#include "common.hpp"

inline vec3i GetGroupSize(int x, int y = 1, int z = 1) {
    constexpr int group_thread_size_x = 16;
    constexpr int group_thread_size_y = 16;
    constexpr int group_thread_size_z = 16;
    const int group_size_x = (x + group_thread_size_x - 1) / group_thread_size_x;
    const int group_size_y = (y + group_thread_size_y - 1) / group_thread_size_y;
    const int group_size_z = (z + group_thread_size_z - 1) / group_thread_size_z;
    return {group_size_x, group_size_y, group_size_z};
}

class TransmittanceGenerator{
  public:
    void initialize() {
        c_shader = program_t::build_from(
            shader_t<GL_COMPUTE_SHADER>::from_file("asset/glsl/Transmittance.comp"));
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

    }
    void generate(const AtmosphereProperties& ap,
                  const Ref<texture2d_t>& trans_lut,
                  const vec2i& lut_size,
                  const vec3f& ground_albedo,
                  int ray_marching_steps,
                  int dir_samples){

    }
    Ref<texture2d_t> getLUT(){

    }
  private:

};

//生成低分辨率的天空纹理图，使用compute shader
class SkyLUTGenerator{
  public:
    void initialize() {

    }

  private:

};

class SkyViewRenderer{
  public:
    void initialize() {

    }

  private:

};


//太阳光的shadow map，不用于spot light和point light
class SunLightShadowRenderer{
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

class VolumeRenderer : public gl_app_t{
public:

    using gl_app_t::gl_app_t;

private:

    void initialize() override {
        // opengl
        GL_EXPR(glEnable(GL_DEPTH_TEST));
        GL_EXPR(glClearColor(0, 0, 0, 0));
        GL_EXPR(glClearDepth(1.0));
    }

	void frame() override {
        handle_events();


        // get camera frustum

        // get intersected volume proxy aabb



    }

	void destroy() override {

    }
private:
    // compute world scene boundary
    void loadScene(){

    }

    void loadTerrain(){

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


};


int main(){
    VolumeRenderer(window_desc_t{
            .size = {1600, 900},
            .title = "VolumeRenderer",
            .resizeable = false,
            .multisamples = 4,
    }).run();
}