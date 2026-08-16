/*
 * Copyright 2026 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.api.core

import org.junit.jupiter.api.Assertions.assertFalse
import org.junit.jupiter.api.Assertions.assertTrue
import org.junit.jupiter.api.Test
import org.junit.jupiter.api.condition.EnabledOnOs
import org.junit.jupiter.api.condition.OS
import org.tensorflow.DeviceSpec
import org.tensorflow.Graph
import org.tensorflow.Session
import org.tensorflow.TensorFlow

/**
 * Pins down which acceleration backends this module can actually reach.
 *
 * These are deliberately *negative* assertions about Apple Metal. Metal acceleration for TensorFlow
 * is delivered by the `tensorflow-metal` PluggableDevice, which is published only as a CPython
 * wheel (`tensorflow_metal-*-cp3xx-macosx_*_arm64.whl`) and is loaded through
 * `TF_LoadPluggableDeviceLibrary`. TensorFlow Java exposes no binding for that call, so there is no
 * supported way to register Metal from the JVM and KotlinDL runs its TensorFlow ops on the CPU.
 *
 * The point of these tests is to fail if that ever stops being true -- for instance after a
 * TensorFlow Java upgrade that adds PluggableDevice support. A failure here is not a bug to be
 * silenced; it is a signal that GPU support should be revisited, and that the notes in README.md
 * and CLAUDE.md need updating.
 */
class AccelerationBackendTest {

    /**
     * A PluggableDevice such as `tensorflow-metal` can only be registered through a binding for the
     * `TF_LoadPluggableDeviceLibrary` C API call. [TensorFlow] offers `loadLibrary` (custom op
     * kernels) and `registerFilesystemPlugin` (filesystem plugins) but nothing for pluggable
     * devices, which is the concrete reason Metal is unreachable from Kotlin/Java.
     */
    @Test
    fun tensorFlowJavaExposesNoPluggableDeviceLoader() {
        val loaderLikeMethods = TensorFlow::class.java.methods
            .map { it.name }
            .filter { it.contains("pluggable", ignoreCase = true) || it.contains("device", ignoreCase = true) }

        assertTrue(
            loaderLikeMethods.isEmpty(),
            "TensorFlow Java now exposes device/pluggable-device entry points $loaderLikeMethods. " +
                    "PluggableDevice support may have landed, so Metal acceleration on macOS should be " +
                    "re-evaluated along with the GPU notes in README.md."
        )
    }

    /**
     * The Metal plugin ships as a Python wheel, so its native library should never appear on the
     * JVM classpath. If it ever does, something is being wired up that these tests do not describe.
     */
    @Test
    fun noMetalPluginIsOnTheClasspath() {
        val loader = AccelerationBackendTest::class.java.classLoader
        val metalArtifacts = listOf(
            "libmetal_plugin.dylib",
            "tensorflow-plugins/libmetal_plugin.dylib",
            "org/tensorflow/native/metal_plugin"
        ).filter { loader.getResource(it) != null }

        assertTrue(
            metalArtifacts.isEmpty(),
            "Found Metal plugin artifacts on the classpath: $metalArtifacts."
        )
    }

    /**
     * TensorFlow Java models device placement with a fixed set of device types. Metal is not one of
     * them; it would surface as a [DeviceSpec.DeviceType.CUSTOM] device only if the plugin were
     * registered, which the previous tests establish it cannot be.
     */
    @Test
    fun deviceTypesDoNotIncludeMetal() {
        val deviceTypes = DeviceSpec.DeviceType.values().map { it.name }
        assertFalse(
            deviceTypes.any { it.contains("METAL", ignoreCase = true) },
            "DeviceSpec.DeviceType now contains a Metal entry: $deviceTypes."
        )
    }

    /**
     * The functional counterpart of the reflection checks: on macOS there is no usable accelerator,
     * so pinning an op to a GPU device must fail. TensorFlow Java publishes a GPU native for
     * `linux-x86_64` only, so this assertion is scoped to macOS.
     */
    @Test
    @EnabledOnOs(OS.MAC)
    fun gpuPlacementIsUnavailableOnMacOs() {
        val gpu = DeviceSpec.newBuilder().deviceType(DeviceSpec.DeviceType.GPU).build()

        val placement = runCatching {
            Graph().use { graph ->
                val tf = org.tensorflow.op.Ops.create(graph).withDevice(gpu)
                val constant = tf.constant(1.0f)
                Session(graph).use { session ->
                    session.runner().fetch(constant).run().use { /* forces placement */ }
                }
            }
        }

        assertFalse(
            placement.isSuccess,
            "A GPU device became available on macOS. TensorFlow Java ships no macOS GPU native and " +
                    "Metal cannot be registered from the JVM, so this is unexpected -- revisit the GPU " +
                    "support notes in README.md and CLAUDE.md."
        )

        // Guard against the assertion above passing for an unrelated reason: the failure must be
        // TensorFlow refusing to place the op, not some incidental error in this test.
        val message = placement.exceptionOrNull()?.message.orEmpty()
        assertTrue(
            message.contains("device", ignoreCase = true),
            "Expected a device placement failure, but got: $message"
        )
    }
}
