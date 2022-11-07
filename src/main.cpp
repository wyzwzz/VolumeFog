#include "common.hpp"
#include <queue>
#include <unordered_set>
#include <cyPoint.h>
#include <cySampleElim.h>
#include <fstream>

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
    static void Destroy(){
        sampler.destroy();
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

void DestroyAllSampler(){
    GL_LinearClampSampler::Destroy();
    GL_LinearRepeatSampler::Destroy();
    GL_NearestClampSampler::Destroy();
    GL_NearestRepeatSampler::Destroy();
}

struct Noise{
    inline static Ref<texture2d_t> BlueNoise;

    inline static vec2i BlueNoiseRes;

    static void Init(){
        BlueNoise = newRef<texture2d_t>();
        BlueNoise->initialize_handle();
        auto bn = image2d_t<color4b>(load_rgba_from_file("assets/bluenoise.png"));
        BlueNoise->initialize_format_and_data(1, GL_RGBA8, bn);
        BlueNoiseRes = bn.size();
    }

    static void Destroy(){
        BlueNoise->destroy();
        BlueNoise.reset();
    }

};


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
    LocalVolumeGeometryInfo() = default;
    LocalVolumeGeometryInfo(const LocalVolumeGeometryInfo&) = default;
    vec3f low; int uid;
    vec3f high; int pad;
    mat4 model;// not consider rotate current so it's just AABB now
};

struct LocalVolumeCube : public LocalVolumeGeometryInfo{
    explicit LocalVolumeCube(const LocalVolumeGeometryInfo& info)
    : LocalVolumeGeometryInfo(info)
    {
        vao.initialize_handle();
        vbo.initialize_handle();
        vec3 vertices[8];
        vertices[0] = low, vertices[1] = vec3(high.x, low.y, low.z);
        vertices[2] = vec3(high.x, high.y, low.z), vertices[3] = vec3(low.x, high.y, low.z);
        vertices[4] = vec3(low.x, low.y, high.z), vertices[5] = vec3(high.x, low.y, high.z);
        vertices[6] = high, vertices[7] = vec3(low.x, high.y, high.z);
        vbo.reinitialize_buffer_data(vertices, 8, GL_STATIC_DRAW);
        static uint32_t indices[24] = {
            0, 1,  1, 2,  2, 3,  3, 0,
            4, 5,  5, 6,  6, 7,  7, 4,
            0, 4,  1, 5,  2, 6,  3, 7
        };
        ebo.initialize_handle();
        ebo.reinitialize_buffer_data(indices, 24, GL_STATIC_DRAW);
        vao.bind_vertex_buffer_to_attrib(attrib_var_t<vec3>(0), vbo, 0);
        vao.enable_attrib(attrib_var_t<vec3>(0));
        vao.bind_index_buffer(ebo);

    }
    vertex_array_t vao;
    vertex_buffer_t<vec3f> vbo;// line mode for debug
    index_buffer_t<uint32_t> ebo;
};

struct LocalVolume{
    std::string desc_name;
    LocalVolumeGeometryInfo info;

    image3d_t<vec4f> vbuffer0;
    image3d_t<vec4f> vbuffer1;
};

Ref<LocalVolume> LoadLocalVolumeFromTexFile(const std::string& filename, const std::string& desc_name){
    std::ifstream fin(filename);
    int width = 1, height = 1, depth = 1;
    fin >> width >> height >> depth;

    const int voxelCount = width * height * depth;
    std::vector<vec4f> data(voxelCount);

    for(int i = 0; i < voxelCount; ++i)
    {
        float x;
        fin >> x;
        data[i] = vec4f(x);
    }
    auto local_vol = newRef<LocalVolume>();
    local_vol->desc_name = desc_name;
    auto tmp = image3d_t<vec4f>(width, height, depth, data.data());
    auto& dst = local_vol->vbuffer0;
    dst.initialize(128, 128, 128);
    for(int i = 0; i < depth; i++){
        for(int j = 0; j < height; j++){
            for(int k = 0; k < width; k++){
                dst.at(k, j, i) = tmp.at(k, j, i) * 0.01f;
            }
        }
    }
    return local_vol;
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

//        linear_clamp_sampler.initialize_handle();
//        linear_clamp_sampler.set_param(GL_TEXTURE_MIN_FILTER, GL_LINEAR);
//        linear_clamp_sampler.set_param(GL_TEXTURE_MAG_FILTER, GL_LINEAR);
//        linear_clamp_sampler.set_param(GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
//        linear_clamp_sampler.set_param(GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
//        linear_clamp_sampler.set_param(GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);


    }
    void update(float exposure, float w_over_h){
        sky_view_params.exposure = exposure;
        sky_view_params.w_over_h = w_over_h;
        sky_view_params_buffer.set_buffer_data(&sky_view_params);
    }
    // call it before bind to default framebuffer and view port
    void render(const vec3f& view_dir, const vec3f& view_right,
                float view_fov_rad,
                const Ref<texture2d_t>& sky_lut,
                const Ref<texture3d_t>& aerial_lut){
        sky_view_per_frame_params.view_dir = view_dir;
        sky_view_per_frame_params.scale = std::tan(0.5f * view_fov_rad);
        sky_view_per_frame_params.view_right = view_right;
        sky_view_per_frame_params_buffer.set_buffer_data(&sky_view_per_frame_params);

        // buffer
        sky_view_params_buffer.bind(0);
        sky_view_per_frame_params_buffer.bind(1);
        // texture
        sky_lut->bind(0);
        aerial_lut->bind(1);
        GL_LinearClampSampler::Bind(0);
        GL_LinearClampSampler::Bind(1);

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

//    sampler_t linear_clamp_sampler;
};


//太阳光或方向光的shadow map，不用于spot light和point light
class DirectionalLightShadowGenerator{
  public:
    void initialize() {
        shader = program_t::build_from(
            shader_t<GL_VERTEX_SHADER>::from_file("assets/glsl/ShadowMap.vert"),
            shader_t<GL_FRAGMENT_SHADER>::from_file("assets/glsl/ShadowMap.frag"));

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
        GL_EXPR(glClearColor(1.f, 1.f, 1.f, 1.f));
        fbo.clear_color_depth_buffer();
        GL_EXPR(glClearColor(0.f, 0.f, 0.f, 0.f));
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
            shader_t<GL_VERTEX_SHADER>::from_file("assets/glsl/GBuffer.vert"),
            shader_t<GL_FRAGMENT_SHADER>::from_file("assets/glsl/GBuffer.frag"));

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

        {
            fbo.initialize_handle();
            rbo.initialize_handle();
            rbo.set_format(GL_DEPTH32F_STENCIL8, res.x, res.y);
            fbo.attach(GL_DEPTH_STENCIL_ATTACHMENT, rbo);
            fbo.attach(GL_COLOR_ATTACHMENT0, *gbuffer0);
            fbo.attach(GL_COLOR_ATTACHMENT1, *gbuffer1);
        }
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


class WireFrameRenderer{
  public:
    void initialize() {
        shader = program_t::build_from(
            shader_t<GL_VERTEX_SHADER>::from_file("assets/glsl/WireFrame.vert"),
            shader_t<GL_FRAGMENT_SHADER>::from_file("assets/glsl/WireFrame.frag"));

        params_buffer.initialize_handle();
        params_buffer.reinitialize_buffer_data(nullptr, GL_STATIC_DRAW);
    }
    void begin(){
        shader.bind();
        GL_EXPR(glPolygonMode(GL_FRONT_AND_BACK, GL_LINE));
        GL_EXPR(glDisable(GL_DEPTH_TEST));
    }

    void draw(const Ref<LocalVolumeCube>& cube,
              const vec4& line_color,
              const mat4& proj_view){
        params.mvp = proj_view * cube->model;
        params.line_color = line_color;
        params_buffer.set_buffer_data(&params);
        params_buffer.bind(0);

        cube->vao.bind();

        GL_EXPR(glDrawElements(GL_LINES, 24, GL_UNSIGNED_INT, nullptr));

        cube->vao.unbind();
    }

    void end(){
        shader.unbind();
        GL_EXPR(glPolygonMode(GL_FRONT_AND_BACK, GL_FILL));
        GL_EXPR(glEnable(GL_DEPTH_TEST));
    }

  private:
    program_t shader;

    struct alignas(16) Params{
        mat4 mvp;
        vec4 line_color;
    }params;
    std140_uniform_block_buffer_t<Params> params_buffer;

};

// tile based defer render
class TerrainRenderer{
  public:
    void initialize() {
        shader = program_t::build_from(
            shader_t<GL_VERTEX_SHADER>::from_file("assets/glsl/Terrain.vert"),
            shader_t<GL_FRAGMENT_SHADER>::from_file("assets/glsl/Terrain.frag"));

        vao.initialize_handle();

        atmos_prop_buffer.initialize_handle();
        atmos_prop_buffer.reinitialize_buffer_data(nullptr, GL_STATIC_DRAW);

        terrain_params_buffer.initialize_handle();
        terrain_params_buffer.reinitialize_buffer_data(nullptr, GL_STATIC_DRAW);
    }

    void resize(const vec2i& res){
        terrain_params.blue_noise_uv_factor = vec2f((float)res.x / Noise::BlueNoiseRes.x,
                                                    (float)res.y / Noise::BlueNoiseRes.y);
        render_res = res;
    }

    void set(const AtmosphereProperties& ap){
        atmos_prop_buffer.set_buffer_data(&ap);
    }

    void update(const vec3& sun_dir, const vec3& sun_radiance, const mat4& sun_proj_view){
        terrain_params.sun_dir = sun_dir;
        terrain_params.sun_theta = std::asin(-sun_dir.y);
        terrain_params.sun_radiance = sun_radiance;
        terrain_params.shadow_proj_view = sun_proj_view;
    }

    void update(float max_aerial_dist, float world_scale, float jitter_radius){
        terrain_params.max_aerial_dist = max_aerial_dist;
        terrain_params.world_scale = world_scale;
        terrain_params.jitter_factor = vec2f(jitter_radius / render_res.x,
                                             jitter_radius / render_res.y);
    }

    void render(const Ref<texture2d_t>& transmittance_lut,
                const Ref<texture3d_t>& froxel_lut,
                const Ref<texture2d_t>& gbuffer0,
                const Ref<texture2d_t>& gbuffer1,
                const Ref<texture2d_t>& shadow_map,
                const vec3f& view_pos,
                const mat4& camera_proj_view){
        static int frame_index = 0;
        terrain_params.view_pos = view_pos;
        terrain_params.camera_proj_view = camera_proj_view;
        terrain_params.frame_index = ++frame_index;
        terrain_params_buffer.set_buffer_data(&terrain_params);

        transmittance_lut->bind(0);
        froxel_lut->bind(1);
        gbuffer0->bind(2);
        gbuffer1->bind(3);
        shadow_map->bind(4);
        Noise::BlueNoise->bind(5);

        GL_LinearClampSampler::Bind(0);
        GL_LinearClampSampler::Bind(1);
        GL_LinearClampSampler::Bind(2);
        GL_NearestRepeatSampler::Bind(3);
        GL_NearestRepeatSampler::Bind(4);
        GL_NearestRepeatSampler::Bind(5);

        atmos_prop_buffer.bind(0);
        terrain_params_buffer.bind(1);

        shader.bind();
        vao.bind();

        GL_EXPR(glDepthFunc(GL_LEQUAL));
        GL_EXPR(glDrawArrays(GL_TRIANGLE_STRIP, 0, 4));
        GL_EXPR(glDepthFunc(GL_LESS));

        vao.unbind();
        shader.unbind();
    }

  private:
    program_t shader;
    vec2i render_res;
    struct alignas(16) TerrainParams{
        vec3f sun_dir; float sun_theta;
        vec3f sun_radiance; float max_aerial_dist = 2000.f;
        vec3f view_pos; float world_scale = 50.f;
        vec2f blue_noise_uv_factor; vec2f jitter_factor;
        mat4 shadow_proj_view;
        mat4 camera_proj_view;
        int frame_index;
    }terrain_params;
    std140_uniform_block_buffer_t<TerrainParams> terrain_params_buffer;
    std140_uniform_block_buffer_t<AtmosphereProperties> atmos_prop_buffer;

    vertex_array_t vao;
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

class VirtualTexMgr{
  public:
    void initialize(int tex_num_, const vec3i& tex_shape_, int l){
        this->tex_num = tex_num_;
        this->tex_shape = tex_shape_;
        this->block_length = l;
        assert(tex_shape_.x % l == 0 && tex_shape_.y % l == 0 && tex_shape_.z % l == 0);

        vec3i size = tex_shape_ / l;
        for(int i = 0; i < tex_num_; i++){
            for(int z = 0; z < size.z; z++){
                for(int y = 0; y < size.y; y++){
                    for(int x = 0; x < size.x; x++){
                        table.emplace_back(x, y, z, i);
                    }
                }
            }
        }

        vpt_buffer = newRef<std140_uniform_block_buffer_t<VPageTable>>();
        vpt_buffer->initialize_handle();
        vpt_buffer->reinitialize_buffer_data(nullptr, GL_STATIC_DRAW);

        for(int i = 0; i < tex_num_; i++){
            auto& tex  = virtual_textures.emplace_back(newRef<texture3d_t>());
            tex->initialize_handle();
            tex->initialize_texture(1, GL_RGBA32F, tex_shape_.x, tex_shape_.y, tex_shape_.z);
        }


        reset();
    }


    void reset(){
        while(!free_blocks.empty()) free_blocks.pop();
        for(auto& x : table) free_blocks.push(x);
        used_vol_set.clear();
        vpt = VPageTable();

    }

    //先暂时只是固定大小的纹理尺寸
    bool upload(const Ref<LocalVolume>& vol){
        if(used_vol_set.count(vol->info.uid)){
            std::cerr << "duplicate upload local volume" << std::endl;
            return false;
        }
        if(free_blocks.size() < 2) return false;
        int uid = vol->info.uid;
        assert(uid >= 0 && uid < 15);
        used_vol_set.insert(uid);

        auto tex_coord = free_blocks.front();
        free_blocks.pop();

        const int pad = 1;
        auto pos = tex_coord.xyz() * block_length;
        vpt.infos[uid].origin0 = vec4(pos.x + pad, pos.y + pad, pos.z + pad, tex_coord.w);
        vpt.infos[uid].shape0 = vec4(block_length - 2 * pad, block_length - 2 * pad, block_length - 2 * pad, 0);

        virtual_textures[tex_coord.w]->set_texture_data(pos.x, pos.y, pos.z,
                                                        block_length, block_length, block_length,
                                                        vol->vbuffer0.get_raw_data());

        tex_coord = free_blocks.front();
        free_blocks.pop();

        pos = tex_coord.xyz() * block_length;
        vpt.infos[uid].origin1 = vec4(pos.x + pad, pos.y + pad, pos.z + pad, tex_coord.w);
        vpt.infos[uid].shape1 = vec4(block_length - 2 * pad, block_length - 2 * pad, block_length - 2 * pad, 0);

        virtual_textures[tex_coord.w]->set_texture_data(pos.x, pos.y, pos.z,
                                                        block_length, block_length, block_length,
                                                        vol->vbuffer1.get_raw_data());

        vpt_buffer->set_buffer_data(&vpt);

        return true;
    }
    auto getVPTBuffer() const {
        return vpt_buffer;
    }
    const auto& getVirtualTextures() const {
        return virtual_textures;
    }
    vec3f get_inv_tex_shape() const{
        return vec3f(1.f / tex_shape.x, 1.f / tex_shape.y, 1.f / tex_shape.z);
    }
  private:
    vec3i tex_shape;
    int tex_num;
    int block_length;
    struct VirtualInfoT{
        vec4 origin0, origin1;
        vec4 shape0, shape1;
    };
    struct alignas(16) VPageTable{
        VirtualInfoT infos[16];
    }vpt;
    Ref<std140_uniform_block_buffer_t<VPageTable>> vpt_buffer;
    std::vector<vec4i> table;
    std::queue<vec4i> free_blocks;
    std::unordered_set<int> used_vol_set;


    std::vector<Ref<texture3d_t>> virtual_textures;

};

//填充froxel属性并计算累积值 包括大气散射、局部雾和全局雾
//体积阴影在mesh渲染处考虑，这里仅考虑体积自遮挡
class AerialLUTGenerator{
  public:
    void initialize() {
        c_fill_shader = program_t::build_from(
            shader_t<GL_COMPUTE_SHADER>::from_file("assets/glsl/FillVolumeMedia.comp"));

        c_calc_shader = program_t::build_from(
            shader_t<GL_COMPUTE_SHADER>::from_file("assets/glsl/CalcFroxel.comp"));


        vbuffer0 = newRef<texture3d_t>();
        vbuffer1 = newRef<texture3d_t>();
        froxel_lut = newRef<texture3d_t>();

        intersect_vol_uid_buffer.initialize_handle();
        intersect_vol_uid_buffer.reinitialize_buffer_data(nullptr, GL_STATIC_DRAW);

        intersect_geometry_buffer.initialize_handle();
        intersect_geometry_buffer.reinitialize_buffer_data(nullptr, GL_STATIC_DRAW);

        aerial_params_buffer.initialize_handle();
        aerial_params_buffer.reinitialize_buffer_data(nullptr, GL_STATIC_DRAW);

        volume_params_buffer.initialize_handle();
        volume_params_buffer.reinitialize_buffer_data(nullptr, GL_STATIC_DRAW);
    }

    void resize(const vec3i& res){
        if(res == vbuffer_res) return;
        vbuffer_res = res;
        vbuffer0->destroy();
        vbuffer1->destroy();

        vbuffer0->initialize_handle();
        vbuffer0->initialize_texture(1, GL_RGBA32F, res.x, res.y, res.z);

        vbuffer1->initialize_handle();
        vbuffer1->initialize_texture(1, GL_RGBA32F, res.x, res.y, res.z);

        froxel_lut->initialize_handle();
        froxel_lut->initialize_texture(1, GL_RGBA32F, res.x, res.y, res.z);

        atmos_prop_buffer.initialize_handle();
        atmos_prop_buffer.reinitialize_buffer_data(nullptr, GL_STATIC_DRAW);

        aerial_params.slice_z_count = res.z;
        volume_params.slice_z_count = res.z;
    }

    void set(const AtmosphereProperties& ap){
        atmos_prop_buffer.set_buffer_data(&ap);
    }

    void prepare(const mat4& proj, const mat4& view, const std::vector<Ref<LocalVolume>>& local_volumes){
        frustum_extf camera_frustum;
        auto proj_view = proj * view;
        extract_frustum_from_matrix(proj_view, camera_frustum, true);
        std::vector<int> view_volumes(1);
        std::unordered_map<int, Ref<LocalVolume>> mp;
        for(auto& local_volume : local_volumes){
            auto box = aabb3f(local_volume->info.low, local_volume->info.high);
            if(get_box_visibility(camera_frustum, box) != BoxVisibility::Invisible){
                view_volumes.push_back(local_volume->info.uid);
                mp[local_volume->info.uid] = local_volume;
            }
        }
        view_volumes[0] = view_volumes.size() - 1;
        std::memset(intersect_volume.uid, 0, sizeof(intersect_volume.uid));
        for(int i = 0; i < view_volumes[0] + 1; i++)
            intersect_volume.uid[i] = view_volumes[i];
        intersect_vol_uid_buffer.set_buffer_data(&intersect_volume);
        LOG_DEBUG("camera view local volume count : {0}", view_volumes[0]);
        std::memset(&intersect_geometry, 0, sizeof(intersect_geometry));
        for(int i = 0; i < view_volumes[0]; i++)
            intersect_geometry.geo[view_volumes[i + 1]] = mp[view_volumes[i + 1]]->info;
        intersect_geometry_buffer.set_buffer_data(&intersect_geometry);

        auto inv_proj_view = proj_view.inverse();

        const vec4f A0 = inv_proj_view * vec4f(-1, 1, 0.2f, 1);
        const vec4f A1 = inv_proj_view * vec4f(-1, 1, 0.5f, 1);

        const vec4f B0 = inv_proj_view * vec4f(1, 1, 0.2f, 1);
        const vec4f B1 = inv_proj_view * vec4f(1, 1, 0.5f, 1);

        const vec4f C0 = inv_proj_view * vec4f(-1, -1, 0.2f, 1);
        const vec4f C1 = inv_proj_view * vec4f(-1, -1, 0.5f, 1);

        const vec4f D0 = inv_proj_view * vec4f(1, -1, 0.2f, 1);
        const vec4f D1 = inv_proj_view * vec4f(1, -1, 0.5f, 1);

        aerial_params.frustum_a = (A1 / A1.w - A0 / A0.w).xyz().normalized();
        aerial_params.frustum_b = (B1 / B1.w - B0 / B0.w).xyz().normalized();
        aerial_params.frustum_c = (C1 / C1.w - C0 / C0.w).xyz().normalized();
        aerial_params.frustum_d = (D1 / D1.w - D0 / D0.w).xyz().normalized();

        volume_params.proj = proj;
        volume_params.inv_proj = proj.inverse();
        volume_params.inv_view = view.inverse();
    }

    void set(const vec3& sun_dir, const mat4& sun_proj_view){
        aerial_params.sun_dir = sun_dir;
        aerial_params.sun_theta = std::asin(-sun_dir.y);
        aerial_params.sun_proj_view = sun_proj_view;
    }
    void set(float max_aerial_dist, bool enable_shadow, int ray_marching_steps){
        aerial_params.max_aerial_dist = max_aerial_dist;
        aerial_params.enable_shadow = static_cast<int>(enable_shadow);
        aerial_params.ray_marching_steps_per_slice = ray_marching_steps;
    }

    void prepare(const VirtualTexMgr& mgr){
        mgr.getVPTBuffer()->bind(2);
        auto& texes = mgr.getVirtualTextures();
        const int unit_base = 2;
        int unit_offset = 0;
        for(auto& tex : texes){
            tex->bind(unit_base + unit_offset++);
        }
        volume_params.inv_virtual_tex_shape = mgr.get_inv_tex_shape();
    }

    void set(bool fill_vol){
        volume_params.fill_vol = fill_vol;
    }

    void generate(const Ref<texture2d_t>& trans_lut,
                  const Ref<texture2d_t>& multi_scat_lut,
                  const Ref<texture2d_t>& shadow_map,
                  const vec3& view_pos){
        static int frame_index = 0;
        aerial_params.view_height = view_pos.y * 50.f;
        aerial_params.view_pos = view_pos;
        aerial_params.frame_index = ++frame_index;
        aerial_params_buffer.set_buffer_data(&aerial_params);

        volume_params.frame_index = frame_index;
        volume_params_buffer.set_buffer_data(&volume_params);



        c_fill_shader.bind();

        intersect_vol_uid_buffer.bind(0);
        intersect_geometry_buffer.bind(1);
        volume_params_buffer.bind(3);

        GL_LinearRepeatSampler::Bind(2);
        GL_LinearRepeatSampler::Bind(3);

        vbuffer0->bind_image(0, 0, GL_WRITE_ONLY, GL_RGBA32F);
        vbuffer1->bind_image(1, 0, GL_WRITE_ONLY, GL_RGBA32F);

        {
            int x = vbuffer_res.x / 4, y = vbuffer_res.y / 4, z = vbuffer_res.z / 4;
            constexpr int group_thread_size_x = 8;
            constexpr int group_thread_size_y = 8;
            constexpr int group_thread_size_z = 8;
            const int group_size_x = (x + group_thread_size_x - 1) / group_thread_size_x;
            const int group_size_y = (y + group_thread_size_y - 1) / group_thread_size_y;
            const int group_size_z = (z + group_thread_size_z - 1) / group_thread_size_z;
            GL_EXPR(glDispatchCompute(group_size_x, group_size_y, group_size_z));
            GL_EXPR(glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT));
        }

        c_fill_shader.unbind();


        c_calc_shader.bind();
        atmos_prop_buffer.bind(0);
        aerial_params_buffer.bind(1);


        trans_lut->bind(0);
        multi_scat_lut->bind(1);
        vbuffer0->bind(2);
        vbuffer1->bind(3);
        shadow_map->bind(4);
        Noise::BlueNoise->bind(5);
        GL_LinearClampSampler::Bind(0);
        GL_LinearClampSampler::Bind(1);
        GL_LinearClampSampler::Bind(2);
        GL_LinearClampSampler::Bind(3);
        GL_NearestClampSampler::Bind(4);
        GL_NearestRepeatSampler::Bind(5);

        froxel_lut->bind_image(0, 0, GL_WRITE_ONLY, GL_RGBA32F);

        {
            auto group_size = GetGroupSize(vbuffer_res.x, vbuffer_res.y, 1);
            GL_EXPR(glDispatchCompute(group_size.x, group_size.y, 1));
            GL_EXPR(glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT));
        }

        c_calc_shader.unbind();

    }


    auto getLUT() const{
        return froxel_lut;
    }

  private:
    program_t c_fill_shader;
    program_t c_calc_shader;

    std140_uniform_block_buffer_t<AtmosphereProperties> atmos_prop_buffer;

    static constexpr int MaxLocalVolumeN = 15;

    // volume uid that intersect with camera frustum
    struct IntersectVolumeUID{
        vec4i uid[16];
    }intersect_volume;
    std140_uniform_block_buffer_t<IntersectVolumeUID> intersect_vol_uid_buffer;

    struct IntersectGeometryInfo{
        LocalVolumeGeometryInfo geo[16];
    }intersect_geometry;
    std140_uniform_block_buffer_t<IntersectGeometryInfo> intersect_geometry_buffer;

    struct alignas(16) VolumeParams{
        vec3 world_origin;
        int slice_z_count;
        vec3 world_shape; int fill_vol = 1;
        vec3 inv_virtual_tex_shape; int frame_index = 0;
        mat4 proj;
        mat4 inv_proj;
        mat4 inv_view;

    }volume_params;
    std140_uniform_block_buffer_t<VolumeParams> volume_params_buffer;


    struct alignas(16) AerialParams{
        vec3 sun_dir;   float sun_theta;
        vec3 frustum_a; float view_height;
        vec3 frustum_b; float max_aerial_dist;
        vec3 frustum_c; int enable_shadow;
        vec3 frustum_d; int ray_marching_steps_per_slice;
        vec3 view_pos;  int slice_z_count;
        mat4 sun_proj_view;
        int frame_index;
    }aerial_params;
    std140_uniform_block_buffer_t<AerialParams> aerial_params_buffer;


    Ref<texture3d_t> vbuffer0;
    Ref<texture3d_t> vbuffer1;
    Ref<texture3d_t> froxel_lut;
    vec3i vbuffer_res;
};

class FroxelAccumulator{
  public:
    void initialize(){
        c_shader = program_t::build_from(
            shader_t<GL_COMPUTE_SHADER>::from_file("assets/glsl/Accumulate.comp"));

        pre_acc = newRef<texture3d_t>();
        cur_acc = newRef<texture3d_t>();

        params_buffer.initialize_handle();
        params_buffer.reinitialize_buffer_data(nullptr, GL_STATIC_DRAW);
    }
    void resize(const vec3i& _res){
        if(res == _res){
            return;
        }
        res = _res;
        params.slice_z_count = res.z;

        pre_acc->destroy();
        cur_acc->destroy();

        pre_acc->initialize_handle();
        cur_acc->initialize_handle();

        pre_acc->initialize_texture(1, GL_RGBA32F, res.x, res.y, res.z);
        cur_acc->initialize_texture(1, GL_RGBA32F, res.x, res.y, res.z);
    }
    void set(float blend_ratio, float camera_far_z){
        params.blend_ratio = blend_ratio;
        params.camera_far_z = camera_far_z;
    }
    void accumulate(const Ref<texture3d_t>& froxel_lut,
                    const mat4& pre_view,
                    const mat4& pre_mvp,
                    const mat4& cur_view,
                    const mat4& cur_proj){
            params.cur_proj = cur_proj;
            params.cur_inv_proj = cur_proj.inverse();
            params.cur_inv_view = cur_view.inverse();
            params.pre_mvp = pre_mvp;
            params.pre_inv_view = pre_view.inverse();
            params_buffer.set_buffer_data(&params);

            params_buffer.bind(0);

            pre_acc->bind(0);
            GL_LinearClampSampler::Bind(0);
            froxel_lut->bind(1);
            GL_LinearClampSampler::Bind(1);
            cur_acc->bind_image(0, 0, GL_WRITE_ONLY, GL_RGBA32F);

            c_shader.bind();
            {
                int x = res.x / 4, y = res.y / 4, z = res.z / 4;
                constexpr int group_thread_size_x = 8;
                constexpr int group_thread_size_y = 8;
                constexpr int group_thread_size_z = 8;
                const int group_size_x = (x + group_thread_size_x - 1) / group_thread_size_x;
                const int group_size_y = (y + group_thread_size_y - 1) / group_thread_size_y;
                const int group_size_z = (z + group_thread_size_z - 1) / group_thread_size_z;
                GL_EXPR(glDispatchCompute(group_size_x, group_size_y, group_size_z));
                GL_EXPR(glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT));
            }
            c_shader.unbind();

            std::swap(pre_acc, cur_acc);
    }
    auto getLUT(){
        return pre_acc;
    }
  private:
    program_t c_shader;
    vec3i res;
    Ref<texture3d_t> pre_acc;
    Ref<texture3d_t> cur_acc;
    struct alignas(16) Params{
        mat4 cur_proj;
        mat4 cur_inv_proj;
        mat4 cur_inv_view;
        mat4 pre_mvp;
        mat4 pre_inv_view;
        float blend_ratio = 0.05;
        int slice_z_count;
        float camera_far_z;
    }params;
    std140_uniform_block_buffer_t<Params> params_buffer;

};

class PostProcessRenderer{
  public:
    void initialize() {
        shader = program_t::build_from(
            shader_t<GL_VERTEX_SHADER>::from_file("assets/glsl/Quad.vert"),
            shader_t<GL_FRAGMENT_SHADER>::from_file("assets/glsl/PostProcess.frag"));

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

        Noise::Init();

        loadTerrain(transform ::translate(0.f, 15.f, 0.f), "assets/terrain.obj", "", "");

        auto [window_w, window_h] = window->get_window_size();

        offscreen_frame.fbo.initialize_handle();
        offscreen_frame.rbo.initialize_handle();
        offscreen_frame.rbo.set_format(GL_DEPTH32F_STENCIL8, window_w, window_h);
        offscreen_frame.fbo.attach(GL_DEPTH_STENCIL_ATTACHMENT, offscreen_frame.rbo);
        offscreen_frame.color = newRef<texture2d_t>();
        offscreen_frame.color->initialize_handle();
        offscreen_frame.color->initialize_texture(1, GL_RGBA32F, window_w, window_h);
        offscreen_frame.fbo.attach(GL_COLOR_ATTACHMENT0, *offscreen_frame.color);
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
        aerial_lut_generator.set(std_unit_atmos_prop);
        aerial_lut_generator.set(sun_dir, _);
        aerial_lut_generator.set(camera.get_far_z() * world_scale, enable_volumetric_shadow, ray_marching_steps_per_slice);

        gbuffer_generator.initialize();
        gbuffer_generator.resize(window->get_window_size());

        dl_shadow_generator.initialize();

        terrain_renderer.initialize();
        terrain_renderer.resize(window->get_window_size());
        terrain_renderer.set(std_unit_atmos_prop);
        terrain_renderer.update(sun_dir, sun_color * sun_intensity, _);
        terrain_renderer.update(camera.get_far_z() * 50.f, 50.f, jitter_radius);

        wireframe_renderer.initialize();


        froxel_accumulator.initialize();
        froxel_accumulator.resize(aerial_lut_res);
        froxel_accumulator.set(blend_ratio, camera.get_far_z());


        post_process_renderer.initialize();

        //camera
        camera.set_position({4.087f, 26.7f, 3.957f});
        camera.set_perspective(CameraFovDegree, 0.1f, 100.f);
        camera.set_direction(0, 0.12);

        // init test local volume
        const int vds = 128;
        vtex_mgr.initialize(2, {512, 512, 512}, vds);
        {
            local_volume_test.vol0 = newRef<LocalVolume>();
            local_volume_test.vol0->desc_name = "test_local_volume0";
            local_volume_test.vol0->info.low = vec3f(0.f, 20.f, 0.f);
            local_volume_test.vol0->info.high = vec3f(7.f, 25.f, 7.f);
            local_volume_test.vol0->info.uid = 1;
            local_volume_test.vol0->info.model = mat4::identity();
            local_volume_test.vol0->vbuffer0 = image3d_t<vec4f>(128, 128, 128, vec4f(0.0001f, 0.001f, 0.f, 0.001));
            local_volume_test.vol0->vbuffer1 = image3d_t<vec4f>(vds, vds, vds, vec4f(0.f, 0.f, 0.f, 0.5f));

            local_volume_test.debug_vol0cube = newRef<LocalVolumeCube>(local_volume_test.vol0->info);

            auto ret = vtex_mgr.upload(local_volume_test.vol0);
            assert(ret);

            local_volume_test.vol1 = newRef<LocalVolume>();
            local_volume_test.vol1->desc_name = "test_local_volume1";
            local_volume_test.vol1->info.low = vec3f(-14.5f, 20.f, -14.f);
            local_volume_test.vol1->info.high = vec3f(-7.5f, 25.f, -7.f);
            local_volume_test.vol1->info.uid = 2;
            local_volume_test.vol1->info.model = mat4::identity();
            local_volume_test.vol1->vbuffer0 = image3d_t<vec4f>(vds, vds, vds, vec4f(0.01f, 0.01f, 0.01f, 0.001f));
            local_volume_test.vol1->vbuffer1 = image3d_t<vec4f>(vds, vds, vds, vec4f(0.f, 0.f, 0.f, -0.5f));

            local_volume_test.debug_vol1cube = newRef<LocalVolumeCube>(local_volume_test.vol1->info);


            ret = vtex_mgr.upload(local_volume_test.vol1);
            assert(ret);

            local_volume_test.vol2 = LoadLocalVolumeFromTexFile("assets/density.txt", "cloud");
            local_volume_test.vol2->info.low = vec3f(10.f, 25.f, 10.f);
            local_volume_test.vol2->info.high = vec3f(30.f, 45.f, 30.f);
            local_volume_test.vol2->info.uid = 3;
            local_volume_test.vol2->info.model = mat4::identity();
            local_volume_test.vol2->vbuffer1 = image3d_t<vec4f>(vds, vds, vds, vec4f(0.f, 0.f, 0.f, 0.f));

            local_volume_test.debug_vol2cube = newRef<LocalVolumeCube>(local_volume_test.vol2->info);

            ret = vtex_mgr.upload(local_volume_test.vol2);
            assert(ret);
        }

    }

	void frame() override {
        auto pre_proj_view = camera.get_view_proj();
        auto pre_view = camera.get_view();
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

                    sky_lut_generator.set(std_unit_atmos_prop);

                    aerial_lut_generator.set(std_unit_atmos_prop);

                    terrain_renderer.set(std_unit_atmos_prop);

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
            auto [sun_dir, _] = getSunLight();
            if(update_sky_lut_params || update_sun){
                sky_lut_generator.update(sun_dir, sun_intensity * sun_color, sky_lut_ray_marching_steps, sky_lut_enable_multi_scattering);
            }

            if(update_sun){
                aerial_lut_generator.set(sun_dir, _);

                terrain_renderer.update(sun_dir, sun_color * sun_intensity, _);
            }

            bool update_aerial = false;
            if(ImGui::TreeNode("Aerial LUT")){
                update_aerial |= ImGui::InputInt("Ray Marching Steps Per Slice", &ray_marching_steps_per_slice);
                if(ImGui::RadioButton("Fill Volume Media", fill_vol)){
                    fill_vol = !fill_vol;
                    aerial_lut_generator.set(fill_vol);
                }
                update_aerial |= ImGui::Checkbox("Enable Volumetric Shadow", &enable_volumetric_shadow);

                ImGui::TreePop();
            }

            if(update_aerial){
                aerial_lut_generator.set(camera.get_far_z() * world_scale, enable_volumetric_shadow, ray_marching_steps_per_slice);
            }

            bool update_terrain = false;
            if(ImGui::TreeNode("Terrain")){
                update_terrain |= ImGui::InputFloat("Jitter Radius", &jitter_radius);

                ImGui::TreePop();
            }
            if(update_terrain){
                terrain_renderer.update(camera.get_far_z() * 50.f, 50.f, jitter_radius);
            }

            if(ImGui::TreeNode("Foxel Accumulate")){
                if(ImGui::SliderFloat("Blend Ratio", &blend_ratio, 0.f, 1.f)){
                    froxel_accumulator.set(blend_ratio, camera.get_far_z());
                }

                ImGui::TreePop();
            }

        }


        // get camera frustum
        auto camera_proj_view = camera.get_view_proj();
        auto camera_view = camera.get_view();
        auto camera_proj = camera.get_proj();
        // get intersected volume proxy aabb

        // render test local volume
        if(1){
            std::vector<Ref<LocalVolume>> visible_local_volumes = {local_volume_test.vol0, local_volume_test.vol1,
            local_volume_test.vol2};
            aerial_lut_generator.prepare(camera_proj, camera_view, visible_local_volumes);

            aerial_lut_generator.prepare(vtex_mgr);

            aerial_lut_generator.generate(trans_generator.getLUT(),
                                          multi_scat_generator.getLUT(),
                                          dl_shadow_generator.getShadowMap(),
                                          camera.get_position()
                                          );
            froxel_accumulator.accumulate(aerial_lut_generator.getLUT(),
                                          pre_view,
                                          pre_proj_view,
                                          camera_view,
                                          camera_proj);
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
        terrain_renderer.render(trans_generator.getLUT(),
                                froxel_accumulator.getLUT(),
                                gbuffer0, gbuffer1,
                                dl_shadow_generator.getShadowMap(),
                                camera.get_position(),
                                camera_proj_view);


        sky_lut_generator.generate(camera.get_position() * 50.f,
                                   trans_generator.getLUT(),
                                   multi_scat_generator.getLUT());

        auto camera_dir = camera.get_xyz_direction();
        const vec3f world_up = {0.f, 1.f, 0.f};

        bindToOffScreenFrame();
        GL_EXPR(glViewport(0, 0, window->get_window_width(), window->get_window_height()));

        sky_view_renderer.render(camera_dir, cross(camera_dir, world_up).normalized(), deg2rad(CameraFovDegree),
                                 sky_lut_generator.getLUT(),
                                 froxel_accumulator.getLUT());


        {
            wireframe_renderer.begin();

            wireframe_renderer.draw(local_volume_test.debug_vol0cube, vec4(1.f, 0.f, 0.f, 1.f), camera_proj_view);

            wireframe_renderer.draw(local_volume_test.debug_vol1cube, vec4(1.f, 1.f, 0.f, 1.f), camera_proj_view);

            wireframe_renderer.end();
        }



        framebuffer_t::bind_to_default();
        framebuffer_t::clear_color_depth_buffer();
        post_process_renderer.draw(offscreen_frame.color);

        ImGui::End();
        ImGui::PopStyleVar();
    }

	void destroy() override {
        DestroyAllSampler();
        Noise::Destroy();
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
        auto view = transform::look_at(sun_dir * 100.f, {0.f, 0.f, 0.f}, {0.f, 1.f, 0.f});
        auto proj = transform::orthographic(-50.f, 50.f, -50.f, 50.f, 1.f, 200.f);
        return std::make_tuple(-sun_dir, proj * view);
    }
    void bindToOffScreenFrame(bool clear = false){
        offscreen_frame.fbo.bind();
        GL_EXPR(glDrawBuffer(GL_COLOR_ATTACHMENT0));
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

    VirtualTexMgr vtex_mgr;

    AerialLUTGenerator aerial_lut_generator;
    vec3i aerial_lut_res = {240, 180, 256};
    int ray_marching_steps_per_slice = 2;
    bool enable_volumetric_shadow = true;
    bool fill_vol = true;

    GBufferGenerator gbuffer_generator;

    DirectionalLightShadowGenerator dl_shadow_generator;

    TerrainRenderer terrain_renderer;
    float jitter_radius = 5.f;

    WireFrameRenderer wireframe_renderer;

    FroxelAccumulator froxel_accumulator;
    float blend_ratio = 0.05f;

    PostProcessRenderer post_process_renderer;

    static constexpr float CameraFovDegree = 60.f;

    bool vsync = true;

    float exposure = 10.f;

    float world_scale = 50.f;

    // test and debug
    struct {
        Ref<LocalVolume> vol0;
        Ref<LocalVolumeCube> debug_vol0cube;
        Ref<LocalVolume> vol1;
        Ref<LocalVolumeCube> debug_vol1cube;

        Ref<LocalVolume> vol2;
        Ref<LocalVolumeCube> debug_vol2cube;
    }local_volume_test;

};


int main(){
    SET_LOG_LEVEL_DEBUG
    VolumeRenderer(window_desc_t{
            .size = {1200, 900},
            .title = "VolumeRenderer",
            .resizeable = false,
            .multisamples = 4,
    }).run();
}