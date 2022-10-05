#pragma once

#include <CGUtils/api.hpp>
#include <CGUtils/model.hpp>
#include <CGUtils/image.hpp>
using namespace wzz::model;
using namespace wzz::gl;

struct alignas(16) AtmosphereProperties {
    vec3f rayleigh_scattering = {5.802f, 13.558f, 33.1f};//10^(-6)m^(-1)
    float rayleigh_density_h = 8.f;//km

    float mie_scattering = 3.996f;
    float mie_asymmetry_g = 0.88f;//scalar
    float mie_absorption = 4.4f;
    float mie_density_h = 1.2f;

    vec3f ozone_absorption = {0.65f, 1.881f, 0.085f};//10^(-6)m^(-1)
    // absolute height = ground_radius + ozone_center_h
    float ozone_center_h = 25;//km
    float ozone_width = 30;//km

    float ground_radius = 6360;//km
    float top_atmosphere_radius = 6460;//km
    float padding = 0;

    // to meter
    AtmosphereProperties toStdUnit() const {
        AtmosphereProperties ret = *this;

        ret.rayleigh_scattering *= 1e-6f;
        ret.rayleigh_density_h *= 1e3f;

        ret.mie_scattering *= 1e-6f;
        ret.mie_absorption *= 1e-6f;
        ret.mie_density_h *= 1e3f;

        ret.ozone_absorption *= 1e-6f;
        ret.ozone_center_h *= 1e3f;
        ret.ozone_width *= 1e3f;

        ret.ground_radius *= 1e3f;
        ret.top_atmosphere_radius *= 1e3f;

        return ret;
    }
};
static_assert(sizeof(AtmosphereProperties) == 64, "");

template <typename T>
using Ref = std::shared_ptr<T>;

template <typename T, typename... Args>
auto newRef(Args&&... args){
    return std::make_shared<T>(std::forward<Args>(args)...);
}