// Objective-C CoreML shim for the qwen-asr ANE matmul offload.
//
// Compiled by build.rs (cc, -fobjc-arc) only on macOS when the `mac-ane`
// feature is on. Exposes a small C ABI consumed by src/mac_ane.rs:
//
//   qwen_ane_create(spec_bytes) -> opaque model handle
//   qwen_ane_run(handle, x, y) -> 0 on success
//   qwen_ane_device(handle, buf) -> writes planned compute device name
//   qwen_ane_free(handle)
//
// The model handle owns a compiled, ANE-targeted MLModel plus reusable
// fp32 input/output MLMultiArrays so per-call allocation does not pollute
// the benchmark timings.

#import <Foundation/Foundation.h>
#import <CoreML/CoreML.h>

@interface QwenAneModel : NSObject
@property(strong) MLModel *model;
@property(strong) MLMultiArray *input;
@property(strong) MLMultiArray *output;
@property(strong) MLDictionaryFeatureProvider *provider;
@property(assign) NSInteger inBytes;
@property(assign) NSInteger outBytes;
@property(strong) NSString *device;
// IO geometry, captured at create, for per-call (re-entrant) MLMultiArray wrapping.
@property(strong) NSArray<NSNumber *> *inShape;
@property(strong) NSArray<NSNumber *> *outShape;
@property(strong) NSArray<NSNumber *> *inStrides;
@property(strong) NSArray<NSNumber *> *outStrides;
@property(assign) MLMultiArrayDataType inType;
@property(assign) MLMultiArrayDataType outType;
@end

@implementation QwenAneModel
@end

static size_t element_size(MLMultiArrayDataType t) {
    switch (t) {
        case MLMultiArrayDataTypeFloat16: return 2;
        case MLMultiArrayDataTypeFloat32: return 4;
        case MLMultiArrayDataTypeDouble:  return 8;
        case MLMultiArrayDataTypeInt32:   return 4;
        default: return 4;
    }
}

// Row-major (C-contiguous) strides, in elements, for a given shape:
// stride[i] = product of shape[i+1..]. Used to wrap caller buffers directly.
static NSArray<NSNumber *> *contiguous_strides(NSArray<NSNumber *> *shape) {
    NSUInteger n = shape.count;
    NSMutableArray<NSNumber *> *strides = [NSMutableArray arrayWithCapacity:n];
    for (NSUInteger i = 0; i < n; i++) [strides addObject:@1];
    NSInteger acc = 1;
    for (NSInteger i = (NSInteger)n - 1; i >= 0; i--) {
        strides[i] = @(acc);
        acc *= shape[i].integerValue;
    }
    return strides;
}

static void copy_string(char *buf, size_t buf_len, NSString *s) {
    if (!buf || buf_len == 0) return;
    const char *utf8 = s ? [s UTF8String] : "";
    strncpy(buf, utf8 ? utf8 : "", buf_len - 1);
    buf[buf_len - 1] = '\0';
}

// Inspect the compute plan to discover where CoreML intends to run the
// model's single innerProduct layer. macOS 14.4+ (we target 26.x).
static NSString *plan_device(NSURL *compiledURL, MLModelConfiguration *config) {
    if (@available(macOS 14.4, *)) {
        __block NSString *result = @"unknown";
        dispatch_semaphore_t sem = dispatch_semaphore_create(0);
        [MLComputePlan loadContentsOfURL:compiledURL
                           configuration:config
                       completionHandler:^(MLComputePlan *plan, NSError *error) {
            (void)error;
            if (plan && plan.modelStructure.neuralNetwork) {
                MLModelStructureNeuralNetwork *nn = plan.modelStructure.neuralNetwork;
                if (nn.layers.count > 0) {
                    MLModelStructureNeuralNetworkLayer *layer = nn.layers.firstObject;
                    MLComputePlanDeviceUsage *usage =
                        [plan computeDeviceUsageForNeuralNetworkLayer:layer];
                    id<MLComputeDeviceProtocol> dev = usage.preferredComputeDevice;
                    if ([dev isKindOfClass:[MLNeuralEngineComputeDevice class]]) {
                        result = @"ANE";
                    } else if ([dev isKindOfClass:[MLGPUComputeDevice class]]) {
                        result = @"GPU";
                    } else if ([dev isKindOfClass:[MLCPUComputeDevice class]]) {
                        result = @"CPU";
                    } else if (dev) {
                        result = NSStringFromClass([dev class]);
                    }
                }
            }
            dispatch_semaphore_signal(sem);
        }];
        dispatch_semaphore_wait(sem, dispatch_time(DISPATCH_TIME_NOW, 30 * NSEC_PER_SEC));
        return result;
    }
    return @"unsupported";
}

void *qwen_ane_create(const uint8_t *spec, size_t spec_len, char *err_buf, size_t err_len) {
    @autoreleasepool {
        NSError *error = nil;

        // Write the protobuf spec to a temp .mlmodel file.
        NSString *tmpDir = NSTemporaryDirectory();
        NSString *stem = [[NSProcessInfo processInfo] globallyUniqueString];
        NSString *modelPath = [tmpDir stringByAppendingPathComponent:
                               [stem stringByAppendingString:@".mlmodel"]];
        NSURL *modelURL = [NSURL fileURLWithPath:modelPath];
        NSData *data = [NSData dataWithBytes:spec length:spec_len];
        if (![data writeToURL:modelURL atomically:YES]) {
            copy_string(err_buf, err_len, @"failed to write temp .mlmodel");
            return NULL;
        }

        // Compile to .mlmodelc.
        NSURL *compiledURL = [MLModel compileModelAtURL:modelURL error:&error];
        if (!compiledURL) {
            copy_string(err_buf, err_len, error.localizedDescription ?: @"compile failed");
            [[NSFileManager defaultManager] removeItemAtURL:modelURL error:nil];
            return NULL;
        }

        // Load targeting CPU + Neural Engine.
        MLModelConfiguration *config = [[MLModelConfiguration alloc] init];
        config.computeUnits = MLComputeUnitsCPUAndNeuralEngine;
        MLModel *model = [MLModel modelWithContentsOfURL:compiledURL
                                           configuration:config
                                                   error:&error];
        if (!model) {
            copy_string(err_buf, err_len, error.localizedDescription ?: @"model load failed");
            return NULL;
        }

        // Read IO shapes from the model description.
        MLFeatureDescription *inDesc = model.modelDescription.inputDescriptionsByName[@"x"];
        MLFeatureDescription *outDesc = model.modelDescription.outputDescriptionsByName[@"y"];
        if (!inDesc.multiArrayConstraint || !outDesc.multiArrayConstraint) {
            copy_string(err_buf, err_len, @"model missing x/y multiarray IO");
            return NULL;
        }
        NSArray<NSNumber *> *inShape = inDesc.multiArrayConstraint.shape;
        NSArray<NSNumber *> *outShape = outDesc.multiArrayConstraint.shape;
        MLMultiArrayDataType inType = inDesc.multiArrayConstraint.dataType;
        MLMultiArrayDataType outType = outDesc.multiArrayConstraint.dataType;

        MLMultiArray *input = [[MLMultiArray alloc] initWithShape:inShape
                                                         dataType:inType
                                                            error:&error];
        MLMultiArray *output = [[MLMultiArray alloc] initWithShape:outShape
                                                          dataType:outType
                                                             error:&error];
        if (!input || !output) {
            copy_string(err_buf, err_len, @"failed to allocate IO multiarrays");
            return NULL;
        }
        MLDictionaryFeatureProvider *provider =
            [[MLDictionaryFeatureProvider alloc] initWithDictionary:@{@"x": input}
                                                              error:&error];
        if (!provider) {
            copy_string(err_buf, err_len, error.localizedDescription ?: @"provider init failed");
            return NULL;
        }

        QwenAneModel *m = [[QwenAneModel alloc] init];
        m.model = model;
        m.input = input;
        m.output = output;
        m.provider = provider;
        m.inBytes = input.count * element_size(inType);
        m.outBytes = output.count * element_size(outType);
        m.device = plan_device(compiledURL, config);

        // Capture IO geometry + contiguous (row-major) strides for the re-entrant
        // per-call path in qwen_ane_run.
        m.inShape = inShape;
        m.outShape = outShape;
        m.inType = inType;
        m.outType = outType;
        m.inStrides = contiguous_strides(inShape);
        m.outStrides = contiguous_strides(outShape);

        // Clean up the source .mlmodel (compiled copy is cached by CoreML).
        [[NSFileManager defaultManager] removeItemAtURL:modelURL error:nil];

        return (__bridge_retained void *)m;
    }
}

// x/y are raw element bytes matching the model's declared IO dtype
// (fp16 by default). x_bytes/y_bytes must equal the model's IO byte sizes.
//
// Re-entrant: this binds the caller's x/y buffers directly into per-call
// MLMultiArrays (initWithDataPointer for the input, outputBackings for the
// output) instead of reusing buffers stored on the shared handle. A single
// compiled MLModel can therefore be driven CONCURRENTLY by many worker threads
// (each with its own x/y) — which is what lets N pipeline lanes saturate the
// ANE while sharing one set of compiled models (one weight copy). MLModel
// predictions are thread-safe; only the old shared IO buffers were not.
int qwen_ane_run(void *handle, const void *x, size_t x_bytes, void *y, size_t y_bytes) {
    @autoreleasepool {
        QwenAneModel *m = (__bridge QwenAneModel *)handle;
        if (!m) return -1;
        if ((NSInteger)x_bytes != m.inBytes || (NSInteger)y_bytes != m.outBytes) return -2;

        NSError *error = nil;

        // Wrap the caller's input buffer (no copy). CoreML only reads it; the
        // no-op deallocator leaves ownership with the caller.
        MLMultiArray *input =
            [[MLMultiArray alloc] initWithDataPointer:(void *)x
                                                shape:m.inShape
                                             dataType:m.inType
                                              strides:m.inStrides
                                          deallocator:^(void *p){ (void)p; }
                                                error:&error];
        if (!input) return -5;
        MLDictionaryFeatureProvider *provider =
            [[MLDictionaryFeatureProvider alloc] initWithDictionary:@{@"x": input} error:&error];
        if (!provider) return -6;

        // Bind the caller's output buffer so CoreML writes results straight into
        // it (no alloc, no post-copy).
        MLMultiArray *output =
            [[MLMultiArray alloc] initWithDataPointer:y
                                                shape:m.outShape
                                             dataType:m.outType
                                              strides:m.outStrides
                                          deallocator:^(void *p){ (void)p; }
                                                error:&error];
        if (!output) return -7;
        MLPredictionOptions *opts = [[MLPredictionOptions alloc] init];
        opts.outputBackings = @{@"y": output};

        id<MLFeatureProvider> out = [m.model predictionFromFeatures:provider
                                                            options:opts
                                                              error:&error];
        if (!out) return -3;
        // With outputBackings honored, results are already in `y`. Guard against
        // a CoreML that ignored the backing (returns its own array) by copying.
        MLFeatureValue *yv = [out featureValueForName:@"y"];
        MLMultiArray *arr = yv.multiArrayValue;
        if (arr && arr.dataPointer != y) {
            if ((NSInteger)(arr.count * element_size(arr.dataType)) != (NSInteger)y_bytes) return -4;
            memcpy(y, arr.dataPointer, y_bytes);
        }
        return 0;
    }
}

int qwen_ane_device(void *handle, char *buf, size_t buf_len) {
    QwenAneModel *m = (__bridge QwenAneModel *)handle;
    if (!m) return -1;
    copy_string(buf, buf_len, m.device ?: @"unknown");
    return 0;
}

void qwen_ane_free(void *handle) {
    if (!handle) return;
    QwenAneModel *m = (__bridge_transfer QwenAneModel *)handle; // ARC releases on scope exit
    (void)m;
}
