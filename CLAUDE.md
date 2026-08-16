# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

KotlinDL — a Keras-inspired high-level deep learning API in Kotlin, published as `org.jetbrains.kotlinx:kotlin-deeplearning-*`. Two independent backends sit behind one shared API surface: **TensorFlow 1.15 Java** (training + inference, desktop JVM only) and **ONNX Runtime** (inference only, JVM + Android).

## Build & test

Toolchain: **Gradle 9.6.1, Kotlin 2.4.10, AGP 9.3.1, JDK 17+ (developed and verified on JDK 25)**. AGP 9 sets the floor at JDK 17; published bytecode targets Java 11, matching the library's documented minimum for consumers.

`impl`, `onnx`, and `visualization` apply `com.android.kotlin.multiplatform.library`. Their JVM/common tasks build fine without an Android SDK, but any `*Android*` task — and therefore a full `./gradlew build` — needs **`ANDROID_HOME`** set (or `sdk.dir` in `local.properties`), with **compileSdk 36** and build-tools 36.0.0 installed.

```bash
export ANDROID_HOME=~/Library/Android/sdk
./gradlew build
```

`gradle.properties` sets `org.gradle.daemon=false`, so every invocation forks a single-use daemon. Pass `--daemon` when iterating locally.

| Command | Scope |
|---|---|
| `./gradlew :tensorflow:test` | The bulk of the suite (338 tests): activations, initializers, layers, plus `api/core/integration` tests that really train models. **Needs an extra flag on Apple Silicon** — see below |
| `./gradlew :impl:jvmTest` | Preprocessing / image conversion — fast, good smoke test (33 tests) |
| `./gradlew :onnx:jvmTest` | ONNX preprocessing + model summary (4 tests) |
| `./gradlew :examples:test` | Integration tests that invoke the `examples` mains; downloads pretrained models, up to 8 GB heap |
| `./gradlew fatJar` | Fat jar over all published modules |
| `./gradlew dokkaGenerate` | API docs (Dokka 2; aggregated output lands in `build/dokka/`) |

### Running the TensorFlow tests on Apple Silicon

`org.tensorflow:libtensorflow_jni:1.15.0` ships natives for `linux-x86_64`, `windows-x86_64` and `darwin-x86_64` only — there is no arm64 build and never will be (the TF 1.x Java line was last released in 2019). On an arm64 JVM every TF-backed test fails with `UnsatisfiedLinkError: Cannot find TensorFlow native library for OS: darwin, architecture: aarch64`.

The workaround is to run the *test JVM* (not Gradle) as x86-64 under Rosetta 2. `gradle/tensorflowTestJvm.gradle` wires this up for `:tensorflow` and `:examples`:

```bash
./gradlew :tensorflow:test -Pkotlindl.testJvm=/path/to/x64-jdk/Contents/Home/bin/java
# or: export KOTLINDL_TEST_JVM=/path/to/x64-jdk/Contents/Home/bin/java
```

Two things are required and both are easy to miss:

1. **The JDK must actually be x86-64.** SDKMAN installs arm64 builds on Apple Silicon, so grab an explicit x64 build (e.g. Temurin `OpenJDK21U-jdk_x64_mac_hotspot`). Verify with `file .../bin/java` → `Mach-O 64-bit executable x86_64`.
2. **AVX must be advertised.** The darwin-x86_64 TensorFlow build is compiled with AVX, and Rosetta does not expose AVX by default, so the process aborts (SIGABRT, exit 134) at library load with `F .../cpu_feature_guard.cc:37] The TensorFlow library was compiled to use AVX instructions, but these aren't available on your machine`. The script sets `ROSETTA_ADVERTISE_AVX=1` on the test task, which fixes it.

On an x86-64 host neither flag is needed. Everything else in the repo — ONNX inference (ONNX Runtime 1.16 ships `osx-aarch64`), preprocessing, and all three Android targets — builds and tests natively on arm64.

Single test: `./gradlew :tensorflow:test --tests "*Conv2DTest"`. Note the task name differs by module type — plain-JVM modules (`tensorflow`, `examples`) use `test`, Kotlin Multiplatform modules (`impl`, `onnx`) use `jvmTest`. The `api` module has no test source set at all.

Tests and examples download datasets and pretrained models from `https://kotlindl.s3.amazonaws.com` into a `cache/` directory relative to the working directory (gitignored). First runs are slow and need network.

Setting the `KOTLIN_DL_RELEASE_VERSION` env var makes `:examples` substitute published Maven artifacts for project dependencies — used to smoke-test a release, see `examples/build.gradle`.

## Module graph

```
api            interfaces + data types only; no TF/ONNX dependency. Plain JVM.
impl           shared implementations: preprocessing ops, image utils, ImageNet/COCO labels. KMP: common/jvm/android.
dataset        Dataset/DataLoader + embedded MNIST/CIFAR/FSDD loaders. KMP plugin but JVM target only.
tensorflow     TF 1.15 training & inference.  → api, impl, dataset.  Plain JVM.
onnx           ONNX Runtime inference.        → api, impl.           KMP: common/jvm/android.
visualization  lets-plot + Swing (jvm), detection overlay views (android). → api, tensorflow.
examples       not published; depends on everything.
gradlePlugin   standalone `DownloadModelsPlugin`: fetches model-hub models into an Android module's res/raw at preBuild.
```

**Package names do not track module names.** The `tensorflow` module's code lives under `org.jetbrains.kotlinx.dl.api.core.*` and `org.jetbrains.kotlinx.dl.api.inference.*` — the same package roots the `api` module uses. So `api/…/api/core/metric/Metrics.kt` (the enum) and `tensorflow/…/api/core/metric/Metric.kt` (the TF-graph metric) are package siblings in different modules. Grep across modules rather than inferring a directory from an import.

## Key abstractions

- **`InferenceModel<R>`** (`api/inference`) — `R` is the backend-native result type (`OrtSession.Result` for ONNX, `TensorResult` for TF). `predict(input, extractResult)` hands the raw result to a lambda so native resources are released immediately after; `resultConverter` converts it to floats/arrays. Retaining the result past the lambda leaks.
- **`FloatData = Pair<FloatArray, TensorShape>`** (`api/core/FloatData.kt`) — the universal tensor carrier at every API boundary, with `.floats` / `.shape` accessors.
- **`Operation<I, O>` + `PreprocessingPipeline`** (`api/preprocessing`) — the preprocessing DSL. Builders like `resize {}`, `crop {}`, `rescale {}`, `normalize {}`, `toFloatArray {}` are extension functions that wrap the receiver in a `PreprocessingPipeline`. They are split jvm/android in `impl` because the image type differs (`BufferedImage` vs `Bitmap`). `getOutputShape` must be implemented so a pipeline's output shape is computable without running it — `PreprocessingFinalShapeTest` enforces this.
- **`ModelHub` / `ModelType<T, U>`** (`api/inference/loaders`) — the model zoo. `T` is the raw model, `U` the "easy API" wrapper returned by `loadPretrainedModel` / `hub[modelType]`. Implemented twice: `TFModelHub`+`TFModels` and `ONNXModelHub`+`ONNXModels`, both downloading into a caller-supplied `cacheDirectory`.
- **`ModelSummary` / `ModelWithSummary`** (`api/summary`) — uniform `printSummary()` / `logSummary()` across both backends.

### TensorFlow training stack

`Layer.build(tf, input, isTraining, numberOfLosses)` appends operations to a static TF graph; it computes nothing. `GraphTrainableModel` owns that graph via `KGraph` (which tracks variables and their initializers) and drives compile → build → fit/evaluate/predict. `Sequential` and `Functional` differ only in `buildLayers()`: Sequential chains layers linearly, Functional resolves each layer's `inboundLayers` (populated by the `layer(prev)` invoke operator). Models own a TF `Graph`/`Session` and are `AutoCloseable` — use `model.use { }`.

Adding a layer type touches several places beyond the `Layer` subclass: `inference/keras/ModelLoader` + `ModelSaver` (Keras JSON round-trip), `WeightMappings` (h5 weight naming), and `TfModelSummary` if it has parameters.

Keras interop is in `tensorflow/…/api/inference/keras/`: `ModelLoader` parses `modelConfig.json` (klaxon), `WeightLoader` reads `weights.h5` (jhdf), `ModelSaver` writes both back.

### ONNX inference

`OnnxInferenceModel` wraps an `OrtSession`. Task-specific models implement `OnnxHighLevelModel<I, R>` — supply `internalModel`, `preprocessing`, and `convert(OrtSession.Result)`, and `predict(input: I): R` comes for free. Follow this when adding a task type: the `*ModelBase` classes for object detection / pose / face alignment live in `commonMain`, and the JVM and Android leaves differ only in input image type.

## Conventions

- **Explicit API mode** (`explicitApiWarning()`) is enabled in `api`, `impl`, `dataset`, `tensorflow`, `onnx` — but not `visualization`. Public declarations need explicit visibility modifiers and explicit return types.
- `jvmTarget = JVM_11` in every module via `kotlin { compilerOptions { } }` (Kotlin 2.x removed `kotlinOptions`), even though the build itself needs JDK 17+. `gradlePlugin` targets 17, the Gradle 9 baseline.
- Every source file opens with the JetBrains Apache 2.0 copyright header — copy it from a neighbouring file.
- Tests use JUnit 5. Classes are named `XxxTest` / `XxxTestSuite`; no backticks or underscores in test names (CONTRIBUTING.md).
- In KMP modules, shared code goes in `commonMain` and diverges in `jvmMain` / `androidMain`. Putting a JVM-only API (`BufferedImage`, `java.io.File`) into `commonMain` breaks the Android target.
- All development happens on `master`. PRs should be linked to an issue, and bug fixes should carry a reproducing test.
- `CHANGELOG.md` records API changes per release with PR links; new public API also warrants a KDoc entry and a note there.
