#include "include/qwen_asr/qwen_asr_plugin.h"

#include <flutter_linux/flutter_linux.h>

// Minimal implementation - FFI handles everything via cargokit
G_DEFINE_TYPE(QwenAsrPlugin, qwen_asr_plugin, G_TYPE_OBJECT)

static void qwen_asr_plugin_class_init(QwenAsrPluginClass* klass) {}

static void qwen_asr_plugin_init(QwenAsrPlugin* self) {}

gboolean qwen_asr_plugin_register_with_registrar(FlPluginRegistrar* registrar) {
  // No-op: all functionality is provided via FFI
  return TRUE;
}
