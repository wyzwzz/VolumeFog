#include "common.hpp"

#include <cyPoint.h>
#include <cySampleElim.h>

inline vec3i GetGroupSize(int x, int y = 1, int z = 1) {
    constexpr int group_thread_size_x = 16;
    constexpr int group_thread_size_y = 16;
    constexpr int group_thread_size_z = 16;
    const int group_size_x = (x + group_thread_size_x - 1) / group_thread_size_x;
    const int group_size_y = (y + group_thread_size_y - 1) / group_thread_size_y;
    const int group_size_z = (z + group_thread_size_z - 1) / group_thread_size_z;
    return {group_size_x, group_size_y, group_size_z};
}

std::vector<vec2f> GetPoissonDiskSamples(int count) {
    std::default_random_engine rng{std::random_device()()};
    std::uniform_real_distribution<float> dis(0, 1);

    std::vector<cy::Point2f> rawPoints;
    for (int i = 0; i < count * 10; ++i) {
        const float u = dis(rng);
        const float v = dis(rng);
        rawPoints.push_back({u, v});
    }

    std::vector<cy::Point2f> outputPoints(count);

    cy::WeightedSampleElimination<cy::Point2f, float, 2> wse;
    wse.SetTiling(true);
    wse.Eliminate(
        rawPoints.data(), rawPoints.size(),
        outputPoints.data(), outputPoints.size());

    std::vector<vec2f> result;
    for (auto &p: outputPoints)
        result.push_back({p.x, p.y});

    return result;
}

namespace {

    struct LocalVolumeResc{
        
    };


}

class TransmittanceGenerator{
  public:

    void initialize() {
        c_shader = program_t::build_from(
            shader_t<GL_COMPUTE_SHADER>::from_file("assets/glsl/Transmittance.comp"));
       atmos_prop_buffer.initialize_handle();
       atmos_prop_buffer.reinitialize_buffer_data(nullptr, GL_STATIC_DRAW);
       trans_lut = newRef<texture2d_t>();
    }

    void generate(const AtmosphereProperties& ap, const vec2i& lut_size){
        atmos_prop_buffer.set_buffer_data(&ap);
        atmos_prop_buffer.bind(0);

        trans_lut->destroy();
        trans_lut->initialize_handle();
        trans_lut->initialize_texture(1, GL_RGBA32F, lut_size.x, lut_size.y);
        trans_lut->bind_image(0, 0, GL_WRITE_ONLY, GL_RGBA32F);

        c_shader.bind();

        auto group_size = GetGroupSize(lut_size.x, lut_size.y);
        GL_EXPR(glDispatchCompute(group_size.x, group_size.y, 1));

        GL_EXPR(glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT));

        c_shader.unbind();
    }

    Ref<texture2d_t> getLUT(){
        return trans_lut;
    }

  private:
    program_t c_shader;
    std140_uniform_block_buffer_t<AtmosphereProperties> atmos_prop_buffer;
    Ref<texture2d_t> trans_lut;
};

class MultiScatteringGenerator{
  public:

    void initialize() {
        c_shader = program_t::build_from(
            shader_t<GL_COMPUTE_SHADER>::from_file("assets/glsl/MultiScattering.comp"));
        ms_params_buffer.initialize_handle();
        ms_params_buffer.reinitialize_buffer_data(nullptr, GL_STATIC_DRAW);
        atmos_prop_buffer.initialize_handle();
        atmos_prop_buffer.reinitialize_buffer_data(nullptr, GL_STATIC_DRAW);

        multi_scat_lut = newRef<texture2d_t>();

        linear_clamp_sampler.initialize_handle();
        linear_clamp_sampler.set_param(GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        linear_clamp_sampler.set_param(GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        linear_clamp_sampler.set_param(GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        linear_clamp_sampler.set_param(GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        linear_clamp_sampler.set_param(GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);
    }

    void generate(const AtmosphereProperties& ap,
                  const Ref<texture2d_t>& trans_lut,
                  const vec2i& lut_size,
                  const vec3f& ground_albedo,
                  int ray_marching_steps,
                  int dir_samples){
        ms_params.ground_albedo = ground_albedo;
        ms_params.ray_marching_steps = ray_marching_steps;
        if(dir_samples != ms_params.dir_sample_count){
            ms_params.dir_sample_count = dir_samples;
            random_samples_buffer.destroy();
            random_samples_buffer.initialize_handle();
            auto samples = GetPoissonDiskSamples(dir_samples);
            random_samples_buffer.initialize_buffer_data(samples.data(), samples.size(), GL_DYNAMIC_STORAGE_BIT);
        }
        assert(ms_params.dir_sample_count > 0);
        ms_params_buffer.set_buffer_data(&ms_params);
        atmos_prop_buffer.set_buffer_data(&ap);

        if(lut_size != prev_lut_size){
            prev_lut_size = lut_size;
            multi_scat_lut->destroy();
            multi_scat_lut->initialize_handle();
            multi_scat_lut->initialize_texture(1, GL_RGBA32F, lut_size.x, lut_size.y);
        }
        // buffer
        atmos_prop_buffer.bind(0);
        ms_params_buffer.bind(1);
        random_samples_buffer.bind(0);
        // image
        multi_scat_lut->bind_image(0, 0, GL_WRITE_ONLY, GL_RGBA32F);
        // texture
        trans_lut->bind(0);
        linear_clamp_sampler.bind(0);

        c_shader.bind();

        auto group_size = GetGroupSize(lut_size.x, lut_size.y);
        GL_EXPR(glDispatchCompute(group_size.x, group_size.y, 1));

        GL_EXPR(glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT));

        c_shader.unbind();
    }

    Ref<texture2d_t> getLUT(){
        return multi_scat_lut;
    }

  private:
    struct alignas(16) MultiScatteringParams{
        vec3f ground_albedo = vec3f(0.3f);
        int dir_sample_count = 0;
        vec3f sun_intensity = vec3f(1.f);
        int ray_marching_steps;
    }ms_params;
    std140_uniform_block_buffer_t<MultiScatteringParams> ms_params_buffer;
    std140_uniform_block_buffer_t<AtmosphereProperties> atmos_prop_buffer;
    storage_buffer_t<vec2f> random_samples_buffer;

    program_t c_shader;
    Ref<texture2d_t> multi_scat_lut;

    sampler_t linear_clamp_sampler;

    vec2i prev_lut_size{0, 0};
};

//生成低分辨率的天空纹理图，使用compute shader
class SkyLUTGenerator{
  public:
    void initialize() {
        c_shader = program_t::build_from(
            shader_t<GL_COMPUTE_SHADER>::from_file("assets/glsl/SkyLUT.comp"));

        sky_params_buffer.initialize_handle();
        sky_params_buffer.reinitialize_buffer_data(nullptr, GL_STATIC_DRAW);
        sky_per_frame_params_buffer.initialize_handle();
        sky_per_frame_params_buffer.reinitialize_buffer_data(nullptr, GL_STATIC_DRAW);
        atmos_prop_buffer.initialize_handle();
        atmos_prop_buffer.reinitialize_buffer_data(nullptr, GL_STATIC_DRAW);

        sky_lut = newRef<texture2d_t>();

        linear_clamp_sampler.initialize_handle();
        linear_clamp_sampler.set_param(GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        linear_clamp_sampler.set_param(GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        linear_clamp_sampler.set_param(GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        linear_clamp_sampler.set_param(GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        linear_clamp_sampler.set_param(GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);
    }
    void resize(const vec2i& lut_size){
        if(lut_size == prev_lut_size) return;
        prev_lut_size = lut_size;
        sky_lut->destroy();
        sky_lut->initialize_handle();
        sky_lut->initialize_texture(1, GL_RGBA32F, lut_size.x, lut_size.y);
    }
    void set(const AtmosphereProperties& ap){
        atmos_prop_buffer.set_buffer_data(&ap);
    }
    void update(const vec3f& sun_dir, const vec3f& sun_radiance, int ray_marching_steps, bool multi_scattering){
        sky_params.sun_dir = sun_dir;
        sky_params.sun_radiance = sun_radiance;
        sky_params.ray_marching_steps = ray_marching_steps;
        sky_params.enable_multi_scattering = static_cast<int>(multi_scattering);
        sky_params_buffer.set_buffer_data(&sky_params);
    }
    void generate(const vec3f& view_pos,
                  const Ref<texture2d_t>& trans_lut,
                  const Ref<texture2d_t>& multi_scat_lut){
        sky_per_frame_params.view_pos = view_pos;
        sky_per_frame_params_buffer.set_buffer_data(&sky_per_frame_params);

        // buffer
        atmos_prop_buffer.bind(0);
        sky_params_buffer.bind(1);
        sky_per_frame_params_buffer.bind(2);
        // texture
        trans_lut->bind(0);
        multi_scat_lut->bind(1);
        linear_clamp_sampler.bind(0);
        linear_clamp_sampler.bind(1);
        // image
        sky_lut->bind_image(0, 0, GL_WRITE_ONLY, GL_RGBA32F);

        c_shader.bind();

        assert(prev_lut_size.x && prev_lut_size.y);
        auto group_size = GetGroupSize(prev_lut_size.x, prev_lut_size.y);
        GL_EXPR(glDispatchCompute(group_size.x, group_size.y, 1));

        GL_EXPR(glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT));

        c_shader.unbind();
    }
    Ref<texture2d_t> getLUT(){
        return sky_lut;
    }

  private:
    program_t c_shader;
    Ref<texture2d_t> sky_lut;
    vec2i prev_lut_size{0, 0};

    struct alignas(16) SkyParams{
        vec3f sun_dir;
        int ray_marching_steps;
        vec3f sun_radiance;
        int enable_multi_scattering;
    }sky_params;
    struct alignas(16) SkyPerFrameParams{
        vec3f view_pos;
    }sky_per_frame_params;
    std140_uniform_block_buffer_t<SkyParams> sky_params_buffer;
    std140_uniform_block_buffer_t<SkyPerFrameParams> sky_per_frame_params_buffer;
    std140_uniform_block_buffer_t<AtmosphereProperties> atmos_prop_buffer;

    sampler_t linear_clamp_sampler;
};

class SkyViewRenderer{
  public:
    void initialize() {
        shader = program_t::build_from(
            shader_t<GL_VERTEX_SHADER>::from_file("assets/glsl/Quad.vert"),
            shader_t<GL_FRAGMENT_SHADER>::from_file("assets/glsl/SkyView.frag"));
        quad_vao.initialize_handle();

        sky_view_params_buffer.initialize_handle();
        sky_view_params_buffer.reinitialize_buffer_data(nullptr, GL_STATIC_DRAW);
        sky_view_per_frame_params_buffer.initialize_handle();
        sky_view_per_frame_params_buffer.reinitialize_buffer_data(nullptr, GL_STATIC_DRAW);

        linear_clamp_sampler.initialize_handle();
        linear_clamp_sampler.set_param(GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        linear_clamp_sampler.set_param(GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        linear_clamp_sampler.set_param(GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        linear_clamp_sampler.set_param(GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        linear_clamp_sampler.set_param(GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);


    }
    void update(float exposure, float w_over_h){
        sky_view_params.exposure = exposure;
        sky_view_params.w_over_h = w_over_h;
        sky_view_params_buffer.set_buffer_data(&sky_view_params);
    }
    // call it before bind to default framebuffer and view port
    void render(const vec3f& view_dir, const vec3f& view_right,
                float view_fov_rad,
                const Ref<texture2d_t>& sky_lut){
        sky_view_per_frame_params.view_dir = view_dir;
        sky_view_per_frame_params.scale = std::tan(0.5f * view_fov_rad);
        sky_view_per_frame_params.view_right = view_right;
        sky_view_per_frame_params_buffer.set_buffer_data(&sky_view_per_frame_params);

        // buffer
        sky_view_params_buffer.bind(0);
        sky_view_per_frame_params_buffer.bind(1);
        // texture
        sky_lut->bind(0);
        linear_clamp_sampler.bind(0);

        shader.bind();
        quad_vao.bind();

        GL_EXPR(glDepthFunc(GL_LEQUAL));
        GL_EXPR(glDrawArrays(GL_TRIANGLE_STRIP, 0, 4));
        GL_EXPR(glDepthFunc(GL_LESS));

        quad_vao.unbind();
        shader.unbind();
    }
  private:
    struct alignas(16) SkyViewParams{
        float exposure;
        float w_over_h;
    }sky_view_params;
    struct alignas(16) SkyViewPerFrameParams{
        vec3f view_dir;
        float scale;
        vec3f view_right;
        int pad;
    }sky_view_per_frame_params;
    std140_uniform_block_buffer_t<SkyViewParams> sky_view_params_buffer;
    std140_uniform_block_buffer_t<SkyViewPerFrameParams> sky_view_per_frame_params_buffer;

    program_t shader;
    vertex_array_t quad_vao;

    sampler_t linear_clamp_sampler;
};


//太阳光或方向光的shadow map，不用于spot light和point light
class DirectionalLightShadowRenderer{
  public:
    void initialize() {

    }

  private:

};

class TerrainRenderer{
  public:
    void initialize() {

    }

  private:

};

//每次只更新部分，第一次耗时较长
class ClipMapGenerator{
  public:
    void initialize(float world_bound, float voxel_size, int lut_size) {
        int level = 0;
        float level_voxel_size =  voxel_size * (1 << level);
        while(level_voxel_size < world_bound){
            int level_lut_size = std::min<int>(lut_size, std::ceil(world_bound / level_voxel_size));
            auto& clip_map = clip_maps.emplace_back(std::make_shared<texture3d_t>());
            clip_map->initialize_handle();
            clip_map->initialize_texture(1, GL_RGBA32F, level_lut_size, level_lut_size, level_lut_size);

            level_voxel_size = voxel_size * (1 << ++level);
        }
        clip.world_bound = world_bound;
        clip.voxel_size = voxel_size;
        clip.levels = level;

        c_shader = program_t::build_from(
            shader_t<GL_COMPUTE_SHADER>::from_file("assets/glsl/ClipMap.comp"));

    }

    void generate(){

    }
    const std::vector<Ref<texture3d_t>>& getLUT(){
        return clip_maps;
    }
  private:
    program_t c_shader;
    std::vector<Ref<texture3d_t>> clip_maps;

    struct{
        float world_bound;
        float voxel_size;
        int levels;
    }clip;
};

//填充froxel属性并计算累积值 包括大气散射、局部雾和全局雾
//体积阴影在mesh渲染处考虑，这里仅考虑体积自遮挡
class AerialLUTGenerator{
  public:
    void initialize() {

    }



  private:
    program_t c_fill_shader;
    program_t c_calc_shader;

};



class VolumeRenderer : public gl_app_t{
public:

    using gl_app_t::gl_app_t;

private:

    void initialize() override {
        // opengl
        GL_EXPR(glEnable(GL_DEPTH_TEST));
        GL_EXPR(glClearColor(0, 0, 0, 0));
        GL_EXPR(glClearDepth(1.0));

        std_unit_atmos_prop = preset_atmos_prop.toStdUnit();
        trans_generator.initialize();
        trans_generator.generate(std_unit_atmos_prop, trans_lut_res);

        multi_scat_generator.initialize();
        multi_scat_generator.generate(std_unit_atmos_prop, trans_generator.getLUT(),
                                      multi_scat_lut_res, vec3f(0.3f), 256, 64);

        auto [sun_dir, _] = getSunLight();

        sky_lut_generator.initialize();
        sky_lut_generator.resize(sky_lut_res);
        sky_lut_generator.set(std_unit_atmos_prop);
        sky_lut_generator.update(sun_dir, sun_intensity * sun_color, sky_lut_ray_marching_steps, sky_lut_enable_multi_scattering);

        sky_view_renderer.initialize();
        sky_view_renderer.update(exposure, window->get_window_w_over_h());

        //camera
        camera.set_position({4.087f, 26.7f, 3.957f});
        camera.set_perspective(CameraFovDegree, 0.1f, 100.f);
        camera.set_direction(0, 0.12);

    }

	void frame() override {
        handle_events();

        ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 7.f);
        if(ImGui::Begin("Settings", nullptr, ImGuiWindowFlags_AlwaysAutoResize)){
            ImGui::Text("Press LCtrl to show/hide cursor");
            ImGui::Text("Use W/A/S/D/Space/LShift to move");
            ImGui::Text("FPS: %.0f", ImGui::GetIO().Framerate);
            if (ImGui::Checkbox("VSync", &vsync)) {
                window->set_vsync(vsync);
            }


            if(ImGui::TreeNode("Atmosphere Properties")){
                bool update = false;
                update |= ImGui::InputFloat("Planet Radius (km)", &preset_atmos_prop.ground_radius, 1.f);
                update |= ImGui::InputFloat("Top Atmosphere Radius (km)",
                                            &preset_atmos_prop.top_atmosphere_radius, 1.f);
                update |= ImGui::InputFloat("Ozone Center Altitude (km)", &preset_atmos_prop.ozone_center_h,
                                            1.f);
                update |= ImGui::InputFloat("Ozone Thickness (km)", &preset_atmos_prop.ozone_width, 1.f);
                update |= ImGui::InputFloat3("Ozone Absorption (10e-6 m^-1)",
                                             &preset_atmos_prop.ozone_absorption.x);

                update |= ImGui::InputFloat("Mie Density Height (km)", &preset_atmos_prop.mie_density_h,
                                            0.01f);
                update |= ImGui::InputFloat("Mie Absorption (10e-6 m^-1)", &preset_atmos_prop.mie_absorption,
                                            0.01f);
                update |= ImGui::InputFloat("Mie Scattering (10e-6 m^-1)", &preset_atmos_prop.mie_scattering,
                                            0.01f);
                update |= ImGui::SliderFloat("Mie Asymmetry G", &preset_atmos_prop.mie_asymmetry_g, -1.f,
                                             1.f);

                update |= ImGui::InputFloat("Rayleigh Density Height (km)",
                                            &preset_atmos_prop.rayleigh_density_h);
                update |= ImGui::InputFloat3("Rayleigh Scattering (10e-6 m^-1)",
                                             &preset_atmos_prop.rayleigh_scattering.x);

                update |= ImGui::InputInt2("Transmittance LUT Size", &trans_lut_res.x);
                update |= ImGui::InputInt2("MultiScattering LUT Size", &multi_scat_lut_res.x);

                if(update){
                    std_unit_atmos_prop = preset_atmos_prop.toStdUnit();

                    trans_generator.generate(std_unit_atmos_prop, trans_lut_res);

                    multi_scat_generator.generate(std_unit_atmos_prop, trans_generator.getLUT(),
                                                  multi_scat_lut_res, vec3f(0.3f), 256, 64);
                }

                ImGui::TreePop();
            }

            bool update_sky_lut_params = false;
            if(ImGui::TreeNode("Sky LUT")){
                if(ImGui::InputInt2("Sky LUT Size", &sky_lut_res.x)){
                    sky_lut_generator.resize(sky_lut_res);
                }
                update_sky_lut_params |= ImGui::InputInt("Sky LUT Ray Marching Steps", &sky_lut_ray_marching_steps);
                update_sky_lut_params |= ImGui::Checkbox("Sky LUT Enable MultiScattering", &sky_lut_enable_multi_scattering);
                ImGui::TreePop();
            }

            bool update_sun = false;
            if(ImGui::TreeNode("Sun")){
                update_sun |= ImGui::SliderFloat("Sun X Degree", &sun_x_degree, 0, 360);
                update_sun |= ImGui::SliderFloat("Sun Y Degree", &sun_y_degree, -10.f, 80);
                update_sun |= ImGui::InputFloat("Sun Intensity", &sun_intensity);
                update_sun |= ImGui::ColorEdit3("Sun Color", &sun_color.x);

                ImGui::TreePop();
            }
            update_sky_lut_params |= update_sun;

            if(update_sky_lut_params){
                auto [sun_dir, _] = getSunLight();
                sky_lut_generator.update(sun_dir, sun_intensity * sun_color, sky_lut_ray_marching_steps, sky_lut_enable_multi_scattering);
            }


        }


        // get camera frustum

        // get intersected volume proxy aabb

        sky_lut_generator.generate(camera.get_position() * 50.f,
                                   trans_generator.getLUT(),
                                   multi_scat_generator.getLUT());


        auto camera_dir = camera.get_xyz_direction();
        const vec3f world_up = {0.f, 1.f, 0.f};

        framebuffer_t::bind_to_default();
        framebuffer_t::clear_color_depth_buffer();
        GL_EXPR(glViewport(0, 0, window->get_window_width(), window->get_window_height()));

        sky_view_renderer.render(camera_dir, cross(camera_dir, world_up).normalized(), deg2rad(CameraFovDegree),
                                 sky_lut_generator.getLUT());



        ImGui::End();
        ImGui::PopStyleVar();
    }

	void destroy() override {

    }
private:
    // compute world scene boundary
    void loadScene(){

    }

    void loadTerrain(){

    }

    std::tuple<vec3f, mat4> getSunLight() {
        float sun_y_rad = wzz::math::deg2rad(sun_y_degree);
        float sun_x_rad = wzz::math::deg2rad(sun_x_degree);
        float sun_dir_y = std::sin(sun_y_rad);
        float sun_dir_x = std::cos(sun_y_rad) * std::cos(sun_x_rad);
        float sun_dir_z = std::cos(sun_y_rad) * std::sin(sun_x_rad);
        vec3f sun_dir = {sun_dir_x, sun_dir_y, sun_dir_z};
        auto view = transform::look_at(sun_dir * 50.f, {0.f, 0.f, 0.f}, {0.f, 1.f, 0.f});
        auto proj = transform::orthographic(-50.f, 50.f, -50.f, 50.f, 1.f, 200.f);
        return std::make_tuple(-sun_dir, proj * view);
    }

private:
    // local volume fog
    struct VirtualTexture{

    };

    // sun light
    struct {
        float sun_x_degree = 0.f;
        float sun_y_degree = 30.f;
        float sun_intensity = 1.f;
        vec3f sun_color = vec3(1.f, 1.f, 1.f);
    };

    // scene
    // terrain
    struct DrawMesh{
        vertex_array_t vao;
        vertex_buffer_t<vertex_t> vbo;
        index_buffer_t<uint32_t> ebo;
        mat4 t;
    };
    std::vector<Ref<DrawMesh>> terrain_meshes;

    AtmosphereProperties preset_atmos_prop;
    AtmosphereProperties std_unit_atmos_prop;


    TransmittanceGenerator trans_generator;
    vec2i trans_lut_res{1024, 256};

    MultiScatteringGenerator multi_scat_generator;
    vec2i multi_scat_lut_res{256, 256};

    SkyLUTGenerator sky_lut_generator;
    vec2i sky_lut_res{512, 256};
    int sky_lut_ray_marching_steps = 40;
    bool sky_lut_enable_multi_scattering = true;

    SkyViewRenderer sky_view_renderer;

    static constexpr float CameraFovDegree = 60.f;

    bool vsync = true;

    float exposure = 10.f;

};


int main(){
    VolumeRenderer(window_desc_t{
            .size = {1600, 900},
            .title = "VolumeRenderer",
            .resizeable = false,
            .multisamples = 4,
    }).run();
}