#include "fcarouge/kalman.hpp"

#include <gst/gst.h>

#include "scope.h"
#include <cassert>
#include <memory>

namespace fcarouge::sample {
namespace {

using pipeline_resource = std::unique_ptr<GstElement, void (*)(GstElement *)>;
using pad_resource = std::unique_ptr<GstPad, decltype(gst_object_unref) *>;
using capabilities_resource =
    std::unique_ptr<GstCaps, decltype(gst_caps_unref) *>;
using message_resource =
    std::unique_ptr<GstMessage, decltype(gst_message_unref) *>;
using loop_resource = std::unique_ptr<GMainLoop, decltype(g_main_loop_unref) *>;
using bus_resource = std::unique_ptr<GstBus, decltype(gst_object_unref) *>;
using watch_resource = sr::unique_resource<guint, decltype(g_source_remove) *>;

void pad_handler(GstElement *source, GstPad *pad, GstElement *sink) {
  static_cast<void>(source);

  const capabilities_resource pad_capabilities{gst_pad_get_current_caps(pad),
                                               gst_caps_unref};
  assert(pad_capabilities && "Failed obtaining the sink pad's capabilities.");

  const auto pad_data{gst_caps_get_structure(pad_capabilities.get(), 0)};
  assert(pad_data && "Failed obtaining the sink pad's capabilities's data.");

  const auto pad_type{gst_structure_get_name(pad_data)};
  assert(pad_type && "Failed obtaining the sink pad's type.");

  if (g_str_has_prefix(pad_type, "video/x-raw")) {
    const pad_resource sink_pad{gst_element_get_static_pad(sink, "sink"),
                                gst_object_unref};
    assert(sink_pad && "Failed obtaining the sink pad.");

    gst_pad_link(pad, sink_pad.get());
  }
}

int bus_call(auto bus, auto message, auto loop) {
  static_cast<void>(bus);

  if (GST_MESSAGE_TYPE(message) == GST_MESSAGE_EOS ||
      GST_MESSAGE_TYPE(message) == GST_MESSAGE_ERROR) {
    g_main_loop_quit(static_cast<GMainLoop *>(loop));
  }

  return 1;
}

[[maybe_unused]] auto kf_video{[] {
  assert(gst_init_check(nullptr, nullptr, nullptr) &&
         "Failed to initialize the GStreamer library.");

  const pipeline_resource pipeline{
      gst_pipeline_new(nullptr), [](auto raw_pipeline) {
        assert(GST_STATE_CHANGE_FAILURE !=
                   gst_element_set_state(raw_pipeline, GST_STATE_NULL) &&
               "The pipeline state could not be cleared.");
        gst_object_unref(raw_pipeline);
      }};
  assert(pipeline && "Failed creation of the pipeline.");

  const loop_resource loop{g_main_loop_new(nullptr, false), g_main_loop_unref};
  assert(loop && "Failed creation of the event loop.");

  const auto uri_decoder_bin{gst_element_factory_make_full(
      "uridecodebin", "uri",
      // "https://www.freedesktop.org/software/gstreamer-sdk/data/media/"
      // "sintel_trailer-480p.webm",
      "file:///home/dev/cpp/kalman/kalman/sample/"
      "roundhay_garden_scene_360p_7fps.mp4",
      nullptr)};
  assert(uri_decoder_bin && "Failed creation of the URI decoder bin element.");

  const auto video_convert{gst_element_factory_make("videoconvert", nullptr)};
  assert(video_convert && "Failed creation of the video sink element.");

  const auto video_sink{gst_element_factory_make("autovideosink", nullptr)};
  assert(video_sink && "Failed creation of the video sink element.");

  gst_bin_add_many(GST_BIN((pipeline.get())), uri_decoder_bin, video_convert,
                   video_sink, nullptr);

  assert(gst_element_link_many(video_convert, video_sink, nullptr) &&
         "Failed linking elements.");

  g_signal_connect(uri_decoder_bin, "pad-added", G_CALLBACK(pad_handler),
                   video_convert);

  const bus_resource bus{gst_element_get_bus(pipeline.get()), gst_object_unref};
  const watch_resource watch_id{
      gst_bus_add_watch(bus.get(), bus_call, loop.get()), g_source_remove};

  assert(GST_STATE_CHANGE_FAILURE !=
             gst_element_set_state(pipeline.get(), GST_STATE_PLAYING) &&
         "The pipeline could not be played.");

  g_main_loop_run(loop.get());

  return 0;
}()};

} // namespace
} // namespace fcarouge::sample
