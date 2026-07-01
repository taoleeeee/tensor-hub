plugins {
    alias(libs.plugins.android.application)
    alias(libs.plugins.kotlin.android)
    alias(libs.plugins.kotlin.compose)
    alias(libs.plugins.kotlin.serialization)
}

android {
    namespace = "com.taoleeeee.tensorhub"
    compileSdk = 35

    defaultConfig {
        applicationId = "com.taoleeeee.tensorhub"
        minSdk = 29
        targetSdk = 35
        versionCode = 2
        versionName = "0.2.0"

        // Only bundle arm64 native libs (Pixel 6 Pro is arm64)
        ndk {
            abiFilters += listOf("arm64-v8a")
        }
    }

    buildTypes {
        release {
            isMinifyEnabled = true
            isShrinkResources = true
            proguardFiles(
                getDefaultProguardFile("proguard-android-optimize.txt"),
                "proguard-rules.pro"
            )
        }
    }

    buildFeatures {
        buildConfig = true
        compose = true
        prefab = true
    }

    externalNativeBuild {
        cmake {
            path = file("src/main/cpp/CMakeLists.txt")
        }
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }

    kotlinOptions {
        jvmTarget = "17"
    }
}

dependencies {
    // AndroidX Core
    implementation(libs.core.ktx)
    implementation(libs.activity.compose)
    implementation(libs.lifecycle.runtime)
    implementation(libs.lifecycle.viewmodel)

    // Compose
    implementation(platform(libs.compose.bom))
    implementation(libs.compose.ui)
    implementation(libs.compose.ui.tooling.preview)
    implementation(libs.compose.material3)
    implementation(libs.compose.material.icons)
    debugImplementation(libs.compose.ui.tooling)

    // LiteRT (TensorFlow Lite successor)
    implementation(libs.litert)
    implementation(libs.litert.gpu)
    implementation(libs.litert.select.tf.ops)  // Required for Prefab native C++ build

    // HTTP Server
    implementation(libs.nanohttpd)

    // Kotlin Coroutines + Serialization
    implementation(libs.coroutines.android)
    implementation(libs.serialization.json)
}
