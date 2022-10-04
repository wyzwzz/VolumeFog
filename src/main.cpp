#include <CGUtils/api.hpp>
#include <CGUtils/model.hpp>
#include <CGUtils/image.hpp>
using namespace wzz::model;
using namespace wzz::gl;

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
    struct VirtualTexture{

    };

};


int main(){
    VolumeRenderer(window_desc_t{
            .size = {1600, 900},
            .title = "VolumeRenderer",
            .resizeable = false,
            .multisamples = 4,
    }).run();
}