#include "common.hpp"

#include <cyPoint.h>
#include <cySampleElim.h>

/**
 * @param BA vertex B' pos minus vertex A' pos
 */
inline vec3f ComputeTangent(const vec3f &BA, const vec3f CA,
                            const vec2f &uvBA, const vec2f &uvCA,
                            const vec3f &normal) {
    const float m00 = uvBA.x, m01 = uvBA.y;
    const float m10 = uvCA.x, m11 = uvCA.y;
    const float det = m00 * m11 - m01 * m10;
    if (std::abs(det) < 0.0001f)
        return wzz::math::tcoord3<float>::from_z(normal).x;
    const float inv_det = 1 / det;
    return (m11 * inv_det * BA - m01 * inv_det * CA).normalized();
}

inline vec3i GetGroupSize(int x, int y = 1, int z = 1) {
    constexpr int group_thread_size_x = 16;
    constexpr int group_thread_size_y = 16;
    constexpr int group_thread_size_z = 16;
    const int group_size_x = (x + group_thread_size_x - 1) / group_thread_size_x;
    const int group_size_y = (y + group_thread_size_y - 1) / group_thread_size_y;
    const int group_size_z = (z + group_thread_size_z - 1) / group_thread_size_z;
    return {group_size_x, group_size_y, group_size_z};
}
template<GLint filter, GLint wrap>
struct GLSampler{
    inline static sampler_t sampler;
    static void Init(){
        sampler.initialize_handle();
        sampler.set_param(GL_TEXTURE_MIN_FILTER, filter);
        sampler.set_param(GL_TEXTURE_MAG_FILTER, filter);
        sampler.set_param(GL_TEXTURE_WRAP_S, wrap);
        sampler.set_param(GL_TEXTURE_WRAP_T, wrap);
        sampler.set_param(GL_TEXTURE_WRAP_R, wrap);
    }
    static void Bind(GLuint binding) {
        sampler.bind(binding);
    }
    static void UnBind(GLuint binding){
        sampler.unbind(binding);
    }
};

using GL_LinearClampSampler = GLSampler<GL_LINEAR, GL_CLAMP_TO_EDGE>;
using GL_LinearRepeatSampler = GLSampler<GL_LINEAR, GL_REPEAT>;
using GL_NearestClampSampler = GLSampler<GL_NEAREST, GL_CLAMP_TO_EDGE>;
using GL_NearestRepeatSampler = GLSampler<GL_NEAREST, GL_REPEAT>;

void InitAllSampler(){
    GL_LinearClampSampler::Init();
    GL_LinearRepeatSampler::Init();
    GL_NearestClampSampler::Init();
    GL_NearestRepeatSampler::Init();
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


struct LocalVolumeGeometryInfo{
    vec3f low; int uid;
    vec3f high; int pad;
    mat4 model;// not consider rotate current so it's just AABB now
};

struct LocalVolumeCube : public LocalVolumeGeometryInfo{
    vertex_array_t vao;
    vertex_buffer_t<vec3f> vbo;// line mode for debug
};

struct LocalVolume{
    std::string desc_name;
    LocalVolumeGeometryInfo info;

    image3d_t<vec4f> vbuffer0;
    image3d_t<vec4f> vbuffer1;
};


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
class DirectionalLightShadowGenerator{
  public:
    void initialize() {
        shader = program_t::build_from(
            shader_t<GL_VERTEX_SHADER>::from_file("asset/glsl/ShadowMap.vert"),
            shader_t<GL_FRAGMENT_SHADER>::from_file("asset/glsl/ShadowMap.frag"));

        fbo.initialize_handle();
        rbo.initialize_handle();
        rbo.set_format(GL_DEPTH32F_STENCIL8, ShadowMapSizeX, ShadowMapSizeY);
        fbo.attach(GL_DEPTH_STENCIL_ATTACHMENT, rbo);
        shadow = newRef<texture2d_t>();
        shadow->initialize_handle();
        shadow->initialize_texture(1, GL_R32F, ShadowMapSizeX, ShadowMapSizeY);
        fbo.attach(GL_COLOR_ATTACHMENT0, *shadow);
        assert(fbo.is_complete());

        transform_buffer.initialize_handle();
        transform_buffer.reinitialize_buffer_data(nullptr, GL_STATIC_DRAW);
    }

    void begin(){
        shader.bind();
        fbo.bind();
        GL_EXPR(glViewport(0, 0, ShadowMapSizeX, ShadowMapSizeY));
        GL_EXPR(glDrawBuffer(GL_COLOR_ATTACHMENT0));
        fbo.clear_color_depth_buffer();
    }

    void generate(const vertex_array_t& vao, int count,
              const mat4& model, const mat4& proj_view){
        transform.model = model;
        transform.mvp = proj_view * model;
        transform_buffer.set_buffer_data(&transform);
        transform_buffer.bind(0);

        vao.bind();
        GL_EXPR(glDrawElements(GL_TRIANGLES, count, GL_UNSIGNED_INT, nullptr));
        vao.unbind();
    }

    void end(){
        fbo.unbind();
        shader.unbind();
    }

    Ref<texture2d_t> getShadowMap() const {
        return shadow;
    }

  private:
    program_t shader;
    static constexpr int ShadowMapSizeX = 4096;
    static constexpr int ShadowMapSizeY = 4096;
    struct{
        framebuffer_t fbo;
        renderbuffer_t rbo;
        Ref<texture2d_t> shadow;
    };
    struct Transform{
        mat4 model;
        mat4 mvp;
    }transform;
    std140_uniform_block_buffer_t<Transform> transform_buffer;

};

class GBufferGenerator{
  public:
    void initialize(){
        shader = program_t::build_from(
            shader_t<GL_VERTEX_SHADER>::from_file("asset/glsl/GBuffer.vert"),
            shader_t<GL_FRAGMENT_SHADER>::from_file("asset/glsl/GBuffer.frag"));

        gbuffer0 = newRef<texture2d_t>();
        gbuffer1 = newRef<texture2d_t>();

        transform_buffer.initialize_handle();
        transform_buffer.reinitialize_buffer_data(nullptr, GL_STATIC_DRAW);
    }

    void resize(const vec2i& res){
        if(res == gbuffer_res) return;
        gbuffer_res = res;

        gbuffer0->destroy();
        gbuffer0->initialize_handle();
        gbuffer0->initialize_texture(1, GL_RGBA32F, res.x, res.y);

        gbuffer1->destroy();
        gbuffer1->initialize_handle();
        gbuffer1->initialize_texture(1, GL_RGBA32F, res.x, res.y);
    }

    void begin(){
        shader.bind();
        fbo.bind();
        GL_EXPR(glViewport(0, 0, gbuffer_res.x, gbuffer_res.y));
        static GLenum draw_buffers[] = {GL_COLOR_ATTACHMENT0, GL_COLOR_ATTACHMENT1};
        GL_EXPR(glDrawBuffers(2, draw_buffers));
        fbo.clear_color_depth_buffer();
    }

    void draw(const vertex_array_t& vao, int count,
              const mat4& model, const mat4& view, const mat4& proj,
              const Ref<texture2d_t>& albedo_map,
              const Ref<texture2d_t>& normal_map){
        transform.model = model;
        transform.view_model = view * model;
        transform.mvp = proj * transform.view_model;
        transform_buffer.set_buffer_data(&transform);
        transform_buffer.bind(0);

        albedo_map->bind(0);
        normal_map->bind(1);

        vao.bind();
        GL_EXPR(glDrawElements(GL_TRIANGLES, count, GL_UNSIGNED_INT, nullptr));
        vao.unbind();
    }

    void end(){
        fbo.unbind();
        shader.unbind();
    }

    auto getGBuffer() const {
        return std::make_pair(gbuffer0, gbuffer1);
    }

  private:
    program_t shader;
    struct{
        // compress, pos(3) + normal(2) + albedo(2) + depth(1) = 2 * rgba
        Ref<texture2d_t> gbuffer0;
        Ref<texture2d_t> gbuffer1;
        vec2i gbuffer_res;

        framebuffer_t fbo;
        renderbuffer_t rbo;
    };
    struct Transform{
        mat4 model;
        mat4 view_model;
        mat4 mvp;
    }transform;
    std140_uniform_block_buffer_t<Transform> transform_buffer;
};

// tile based defer render
class TerrainRenderer{
  public:
    void initialize() {
        shader = program_t::build_from(
            shader_t<GL_VERTEX_SHADER>::from_file("asset/glsl/Terrain.vert"),
            shader_t<GL_FRAGMENT_SHADER>::from_file("asset/glsl/Terrain.frag"));

    }

    // using default framebuffer
    void render(const Ref<texture2d_t>& gbuffer0,
                const Ref<texture2d_t>& gbuffer1){

    }

  private:
    program_t shader;
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

        vbuffer0 = newRef<texture3d_t>();
        vbuffer1 = newRef<texture3d_t>();


        intersect_vol_uid_buffer.initialize_handle();
        intersect_vol_uid_buffer.initialize_buffer_data(nullptr, MaxLocalVolumeN + 1, GL_DYNAMIC_STORAGE_BIT);
    }
    void resize(const vec3i& res){
        if(res == vbuffer_res) return;
        vbuffer_res = res;
        vbuffer0->destroy();
        vbuffer1->destroy();

        vbuffer0->initialize_handle();
        vbuffer0->initialize_texture(1, GL_RGBA16F, res.x, res.y, res.z);

        vbuffer1->initialize_handle();
        vbuffer1->initialize_texture(1, GL_RGBA16F, res.x, res.y, res.z);

    }

    void prepare(const mat4& proj_view, const std::vector<Ref<LocalVolume>>& local_volumes){
        frustum_extf camera_frustum;
        extract_frustum_from_matrix(proj_view, camera_frustum, true);
        std::vector<int> view_volumes(1);
        for(auto& local_volume : local_volumes){
            auto box = aabb3f(local_volume->info.low, local_volume->info.high);
            if(get_box_visibility(camera_frustum, box) != BoxVisibility::Invisible){
                view_volumes.push_back(local_volume->info.uid);
            }
        }
        view_volumes[0] = view_volumes.size() - 1;
        intersect_vol_uid_buffer.set_buffer_data(view_volumes.data(), 0, view_volumes.size() * sizeof(int));
        LOG_DEBUG("camera view local volume count : ", view_volumes[0]);



    }

    void generate(){

    }


    auto getLUT() const{
        return std::make_pair(vbuffer0, vbuffer1);
    }

  private:
    program_t c_fill_shader;
    program_t c_calc_shader;

    static constexpr int MaxLocalVolumeN = 15;

    // volume uid that intersect with camera frustum
    storage_buffer_t<int> intersect_vol_uid_buffer;


    Ref<texture3d_t> vbuffer0;
    Ref<texture3d_t> vbuffer1;
    vec3i vbuffer_res;
};

class PostProcessRenderer{
  public:
    void initialize() {
        shader = program_t::build_from(
            shader_t<GL_VERTEX_SHADER>::from_file("asset/glsl/Quad.vert"),
            shader_t<GL_FRAGMENT_SHADER>::from_file("asset/glsl/PostProcess.frag"));

        vao.initialize_handle();
    }

    // must bind to default framebuffer first
    void draw(const Ref<texture2d_t>& color){
        shader.bind();
        vao.bind();

        color->bind(0);
        GL_NearestRepeatSampler::Bind(0);

        GL_EXPR(glDepthFunc(GL_LEQUAL));
        GL_EXPR(glDrawArrays(GL_TRIANGLE_STRIP, 0, 4));
        GL_EXPR(glDepthFunc(GL_LESS));

        vao.unbind();
        shader.unbind();
    }

  private:
    program_t shader;
    vertex_array_t vao;
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

        InitAllSampler();

        loadTerrain(mat4::identity(), "assets/terrain.obj", "", "assets/normal.jpg");

        auto [window_w, window_h] = window->get_window_size();

        offscreen_frame.fbo.initialize_handle();
        offscreen_frame.rbo.initialize_handle();
        offscreen_frame.rbo.set_format(GL_DEPTH32F_STENCIL8, window_w, window_h);
        offscreen_frame.fbo.attach(GL_DEPTH_STENCIL_ATTACHMENT, offscreen_frame.rbo);
        offscreen_frame.color = newRef<texture2d_t>();
        offscreen_frame.color->initialize_handle();
        offscreen_frame.color->initialize_texture(1, GL_RGBA8, window_w, window_h);
        assert(offscreen_frame.fbo.is_complete());

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

        aerial_lut_generator.initialize();
        aerial_lut_generator.resize(aerial_lut_res);

        gbuffer_generator.initialize();
        gbuffer_generator.resize(window->get_window_size());

        dl_shadow_generator.initialize();

        terrain_renderer.initialize();


        post_process_renderer.initialize();

        //camera
        camera.set_position({4.087f, 26.7f, 3.957f});
        camera.set_perspective(CameraFovDegree, 0.1f, 100.f);
        camera.set_direction(0, 0.12);

        // init test local volume
        {
            local_volume_test.vol0->desc_name = "test_local_volume0";
            local_volume_test.vol0->info.low = vec3f(0.f, 0.f, 0.f);
            local_volume_test.vol0->info.high = vec3f(1.f, 1.f, 1.f);
            local_volume_test.vol0->info.uid = 1;
            local_volume_test.vol0->info.model = mat4::identity();
            local_volume_test.vol0->vbuffer0 = image3d_t<vec4f>(128, 128, 128, vec4f(1.f, 1.f, 0.f, 1.f));


            local_volume_test.vol1->desc_name = "test_local_volume1";
            local_volume_test.vol1->info.low = vec3f(1.5f, 0.f, 0.f);
            local_volume_test.vol1->info.high = vec3f(2.5f, 1.f, 1.f);
            local_volume_test.vol1->info.uid = 2;
            local_volume_test.vol1->info.model = mat4::identity();


        }
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
        auto camera_proj_view = camera.get_view_proj();
        auto camera_view = camera.get_view();
        auto camera_proj = camera.get_proj();
        // get intersected volume proxy aabb

        // render test local volume
        {
            std::vector<Ref<LocalVolume>> visible_local_volumes = {local_volume_test.vol0, local_volume_test.vol1};
            aerial_lut_generator.prepare(camera_proj_view, visible_local_volumes);

            aerial_lut_generator.generate();


        }

        auto [sun_dir, sun_proj_view] = getSunLight();

        // shadow map

        dl_shadow_generator.begin();
        for(auto& mesh : terrain_meshes){
            dl_shadow_generator.generate(mesh->vao, mesh->ebo.index_count(),
                                         mesh->t, sun_proj_view);
        }
        dl_shadow_generator.end();

        // render terrian

        gbuffer_generator.begin();
        for(auto& mesh : terrain_meshes){
            gbuffer_generator.draw(mesh->vao, mesh->ebo.index_count(),
                                   mesh->t, camera_view, camera_proj,
                                   mesh->albedo_map, mesh->normal_map);
        }
        gbuffer_generator.end();
        auto [gbuffer0, gbuffer1] = gbuffer_generator.getGBuffer();


        // todo
        bindToOffScreenFrame(true);
        terrain_renderer.render(gbuffer0, gbuffer1);



        sky_lut_generator.generate(camera.get_position() * 50.f,
                                   trans_generator.getLUT(),
                                   multi_scat_generator.getLUT());


        auto camera_dir = camera.get_xyz_direction();
        const vec3f world_up = {0.f, 1.f, 0.f};

//        framebuffer_t::bind_to_default();
//        framebuffer_t::clear_color_depth_buffer();
        bindToOffScreenFrame();
        GL_EXPR(glViewport(0, 0, window->get_window_width(), window->get_window_height()));

        sky_view_renderer.render(camera_dir, cross(camera_dir, world_up).normalized(), deg2rad(CameraFovDegree),
                                 sky_lut_generator.getLUT());


        framebuffer_t::bind_to_default();
        framebuffer_t::clear_color_depth_buffer();
        post_process_renderer.draw(offscreen_frame.color);

        ImGui::End();
        ImGui::PopStyleVar();
    }

	void destroy() override {

    }
private:
    // compute world scene boundary
    void loadScene(){

    }

    void loadTerrain(const mat4& local_to_world, std::string filename, std::string albedo_filename = "", std::string normal_filename = ""){
        auto model = load_model_from_file(filename);
        image2d_t<color3b> albedo = albedo_filename.empty() ? image2d_t<color3b>(2, 2, to_color3b(color3f(0.3f))) : image2d_t<color3b>(load_rgb_from_file(albedo_filename));
        image2d_t<color3b> normal = normal_filename.empty() ? image2d_t<color3b>(2, 2, to_color3b(color3f(0.5f, 0.5f, 1.f))) : image2d_t<color3b>(load_rgb_from_file(normal_filename));

        auto albedo_map = newRef<texture2d_t>();
        albedo_map->initialize_handle();
        albedo_map->initialize_format_and_data(1, GL_RGB8, albedo);
        auto normal_map = newRef<texture2d_t>();
        normal_map->initialize_handle();
        normal_map->initialize_format_and_data(1, GL_RGB8, normal);
        for(auto& mesh : model->meshes){
            auto& draw_mesh = terrain_meshes.emplace_back(newRef<DrawMesh>());
            // todo
            draw_mesh->albedo_map = albedo_map;
            draw_mesh->normal_map = normal_map;

            draw_mesh->t = local_to_world;

            draw_mesh->vao.initialize_handle();
            draw_mesh->vbo.initialize_handle();
            draw_mesh->ebo.initialize_handle();

            std::vector<Vertex> vertices;
            int n = mesh.indices.size() / 3;
            auto &indices = mesh.indices;
            vertices.reserve(mesh.vertices.size());
            for (int i = 0; i < n; i++) {
                const auto &A = mesh.vertices[indices[3 * i]];
                const auto &B = mesh.vertices[indices[3 * i + 1]];
                const auto &C = mesh.vertices[indices[3 * i + 2]];
                const auto BA = B.pos - A.pos;
                const auto CA = C.pos - A.pos;
                const auto uvBA = B.tex_coord - A.tex_coord;
                const auto uvCA = C.tex_coord - A.tex_coord;
                for (int j = 0; j < 3; j++) {
                    const auto &v = mesh.vertices[indices[3 * i + j]];
                    vertices.push_back({
                        v.pos, v.normal, ComputeTangent(BA, CA, uvBA, uvCA, v.normal),
                        v.tex_coord});
                }
            }
            assert(vertices.size() == mesh.vertices.size());
            draw_mesh->vbo.reinitialize_buffer_data(vertices.data(), vertices.size(), GL_STATIC_DRAW);
            draw_mesh->ebo.reinitialize_buffer_data(indices.data(), indices.size(), GL_STATIC_DRAW);
            draw_mesh->vao.bind_vertex_buffer_to_attrib(attrib_var_t<vec3f>(0), draw_mesh->vbo, &Vertex::pos, 0);
            draw_mesh->vao.bind_vertex_buffer_to_attrib(attrib_var_t<vec3f>(1), draw_mesh->vbo, &Vertex::normal, 1);
            draw_mesh->vao.bind_vertex_buffer_to_attrib(attrib_var_t<vec3f>(2), draw_mesh->vbo, &Vertex::tangent, 2);
            draw_mesh->vao.bind_vertex_buffer_to_attrib(attrib_var_t<vec2f>(3), draw_mesh->vbo, &Vertex::tex_coord,3);
            draw_mesh->vao.enable_attrib(attrib_var_t<vec3f>(0));
            draw_mesh->vao.enable_attrib(attrib_var_t<vec3f>(1));
            draw_mesh->vao.enable_attrib(attrib_var_t<vec3f>(2));
            draw_mesh->vao.enable_attrib(attrib_var_t<vec2f>(3));
            draw_mesh->vao.bind_index_buffer(draw_mesh->ebo);
        }
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
    void bindToOffScreenFrame(bool clear = false){
        offscreen_frame.fbo.bind();
        GL_EXPR(glViewport(0, 0, window->get_window_width(), window->get_window_height()));
        if(clear){
            GL_EXPR(glClear(GL_DEPTH_BUFFER_BIT | GL_COLOR_BUFFER_BIT));
        }
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
    struct Vertex{
        vec3f pos;
        vec3f normal;
        vec3f tangent;
        vec2f tex_coord;
    };
    struct DrawMesh{
        vertex_array_t vao;
        vertex_buffer_t<Vertex> vbo;
        index_buffer_t<uint32_t> ebo;
        Ref<texture2d_t> albedo_map;
        Ref<texture2d_t> normal_map;
        mat4 t;
    };
    std::vector<Ref<DrawMesh>> terrain_meshes;
    Ref<texture2d_t> terrain_albedo_map;
    Ref<texture2d_t> terrain_normal_map;

    AtmosphereProperties preset_atmos_prop;
    AtmosphereProperties std_unit_atmos_prop;


    // render to off-screen framebuffer before finally post-process
    struct{
        framebuffer_t fbo;
        renderbuffer_t rbo;
        Ref<texture2d_t> color;
    }offscreen_frame;

    TransmittanceGenerator trans_generator;
    vec2i trans_lut_res{1024, 256};

    MultiScatteringGenerator multi_scat_generator;
    vec2i multi_scat_lut_res{256, 256};

    SkyLUTGenerator sky_lut_generator;
    vec2i sky_lut_res{512, 256};
    int sky_lut_ray_marching_steps = 40;
    bool sky_lut_enable_multi_scattering = true;

    SkyViewRenderer sky_view_renderer;

    AerialLUTGenerator aerial_lut_generator;
    vec3i aerial_lut_res = {200, 150, 32};

    GBufferGenerator gbuffer_generator;

    DirectionalLightShadowGenerator dl_shadow_generator;

    TerrainRenderer terrain_renderer;



    PostProcessRenderer post_process_renderer;

    static constexpr float CameraFovDegree = 60.f;

    bool vsync = true;

    float exposure = 10.f;


    // test and debug
    struct {
        Ref<LocalVolume> vol0;
        Ref<LocalVolume> vol1;
    }local_volume_test;

};


int main(){
    VolumeRenderer(window_desc_t{
            .size = {1600, 900},
            .title = "VolumeRenderer",
            .resizeable = false,
            .multisamples = 4,
    }).run();
}