---
title: The SOG Format
sidebar_label: SOG
---

**SOG (Spatially Ordered Gaussians)** is a compact, compressed container format specifically designed for 3D Gaussian Splat data in real-time neural rendering applications. It achieves high compression through quantization techniques, typically yielding files **15–20× smaller** than equivalent PLY files while maintaining visual quality suitable for web delivery and interactive applications.

## What is the SOG Format?

SOG was developed as a runtime-optimized alternative to the PLY format for 3D Gaussian Splatting. While PLY serves excellently as an uncompressed interchange and archival format, SOG addresses the performance requirements of real-time rendering, particularly for web-based applications where file size and loading speed are critical.

### Key Characteristics

- **Lossy Compression**: Uses quantization to reduce precision while preserving visual fidelity
- **Web-Optimized**: Designed for fast loading and streaming over networks
- **GPU-Friendly**: Structured for efficient GPU processing and rendering
- **Compact Storage**: Significant reduction in file size compared to source PLY files

## Why SOG Exists

3D Gaussian Splatting has revolutionized real-time neural rendering by representing scenes as collections of 3D Gaussians rather than traditional meshes. However, the PLY format used during training and processing contains full-precision floating-point data that results in large file sizes (often hundreds of megabytes to gigabytes).

For real-time applications, especially web-based viewers, these large files present significant challenges:

- **Network Transfer**: Slow download times for users
- **Storage Costs**: Expensive CDN and hosting fees
- **Loading Performance**: Long initialization times before rendering can begin
- **Memory Usage**: High memory requirements on client devices

SOG solves these problems by applying carefully designed compression techniques that maintain the visual quality of the original scene while dramatically reducing file size and improving loading performance.

## Benefits Over PLY

| Aspect | PLY | SOG |
|--------|-----|-----|
| **File Size** | Large (50MB - several GB) | Small (15-20× compression ratio) |
| **Quality** | Lossless | Lossy but visually optimized |
| **Loading Speed** | Slow | Fast |
| **Network Transfer** | Impractical for web | Optimized for web delivery |
| **Memory Usage** | High | Reduced |
| **Use Case** | Training, editing, archival | Runtime, delivery, web apps |

## Compression Approach

SOG achieves its impressive compression ratios through several sophisticated techniques:

### Quantization
- **Position Encoding**: 16-bit quantization of log-transformed 3D positions
- **Orientation Encoding**: 26-bit "smallest-three" quaternion compression
- **Scale Encoding**: Codebook-based quantization of scale values
- **Color Encoding**: Quantized spherical harmonics coefficients

### Lossless Components
- **WebP Compression**: All data stored as WebP images for additional lossless compression
- **Efficient Packing**: Multiple properties stored in RGBA channels
- **Metadata Optimization**: Compact JSON metadata with shared codebooks

### Visual Quality Preservation
- **Perceptually-Based**: Compression designed to minimize visible artifacts
- **Adaptive Quantization**: Different precision levels for different properties
- **Color Space Awareness**: Proper handling of linear vs sRGB color spaces

## Use Cases

### Web Applications
SOG is ideal for web-based 3D viewers where fast loading and small file sizes are essential for user experience.

### Real-Time Rendering
The compressed format enables efficient GPU processing and rendering of complex scenes.

### Content Delivery Networks
Small file sizes make SOG practical for global distribution through CDNs.

### Mobile Applications
Reduced memory requirements and fast loading make SOG suitable for mobile 3D applications.

## File Structure Overview

A SOG dataset consists of a metadata file (`meta.json`) plus a series of WebP images containing the compressed Gaussian properties. This structure allows for efficient streaming and progressive loading.

### Core Components
- **Metadata**: JSON file describing the dataset and compression parameters
- **Position Data**: Quantized 3D positions stored across two WebP images
- **Orientation Data**: Compressed quaternion rotations
- **Scale Data**: Codebook-quantized size information
- **Color Data**: Spherical harmonics coefficients for view-dependent coloring
- **Optional Higher-Order SH**: Additional frequency bands for complex lighting

### Bundled vs Multi-File
SOG supports both multi-file layouts (for development) and single ZIP archives (for distribution), providing flexibility across different deployment scenarios.

---

# Format Specification

This document serves as the complete technical specification for the SOG format.

## 1. File set

A SOG dataset is a set of images plus a metadata file:

| File                 | Purpose                             | Channels (8-bit) |
| -------------------- | ----------------------------------- | ---------------- |
| `meta.json`          | Scene metadata and filenames        | —                |
| `means_l.webp`       | Positions – lower 8 bits (RGB)      | R,G,B            |
| `means_u.webp`       | Positions – upper 8 bits (RGB)      | R,G,B            |
| `quats.webp`         | Orientation – compressed quaternion | R,G,B,A          |
| `scales.webp`        | Per-axis sizes via codebook         | R,G,B            |
| `sh0.webp`           | Base color (DC) + opacity           | R,G,B,A          |
| `shN_labels.webp`    | Indices into SH palette (optional)  | R,G              |
| `shN_centroids.webp` | SH palette coefficients (optional)  | RGBA             |

:::note[Image formats]

* By default, images **should** be **lossless WebP** to preserve quantized values exactly.
* Each property in `meta.json` names its file, so other 8-bit RGBA-capable formats **may** be used.
* Do not use lossy encodings for these assets as lossy compression will corrupt values and can produce visible/structural artifacts.

:::

### 1.1 Image dimensions & indexing

All per-Gaussian properties are co-located: the same pixel (x, y) across all property images (except shN_centroids) belongs to the same Gaussian.

* Pixels are laid out **row-major**, origin at the **top-left**.
* For image width `W` and height `H`, the number of addressable Gaussians is `W*H`.
* `meta.count` **must** be `<= W*H`. Any trailing pixels are ignored.

**Indexing math (zero-based):**

* From index to pixel:
  `x = i % W`, `y = floor(i / W)`
* From pixel to index:
  `i = x + y * W`

### 1.2 Coordinate system

Right-handed:

* **x:** right
* **y:** up
* **z:** back (i.e., −z is “forward” in camera-looking-down −z conventions)

### 1.3 Bundled variant

A bundled SOG is a ZIP of the files above. Readers **should** accept either layout:

* **Multi-file directory** (recommended during authoring)
* **Single archive** (e.g., `scene.sog`) containing the same files at the archive root

Readers **must** unzip and then resolve files using `meta.json` exactly as for the multi-file version.

---

## 2. `meta.json`

The `meta.json` file uses the following JSON schema structure:

```json
{
  "version": 2,              // File format version (int)
  "count": 187543,           // Number of gaussians (<= W*H of the images) (int)

  "means": {
    // Ranges for decoding log-transformed positions (see §3.1)
    "mins": [-2.10, -1.75, -2.40],   // min of nx,ny,nz (log-domain) (float[3])
    "maxs": [ 2.05,  2.25,  1.90],   // max of nx,ny,nz (log-domain) (float[3])
    "files": ["means_l.webp", "means_u.webp"]
  },

  "scales": {
    "codebook": [/* array of 256 floats */],    // Quantization codebook for scale values (float[])
    "files": ["scales.webp"]
  },

  "quats": {
    "files": ["quats.webp"] // Quaternion orientation data
  },

  "sh0": {
    "codebook": [/* array of 256 floats */],    // Quantization codebook for DC SH coefficients (float[])
    "files": ["sh0.webp"]
  },

  // Present only if higher-order SH coefficients exist
  "shN": {
    "count": 128,         // Palette size (up to 65536) (int)
    "bands": 3,           // Number of SH bands (1..3). DC (=band 1) lives in sh0 (int)
    "codebook": [/* array of 256 floats */],    // Shared codebook for AC coefficients (float[])
    "files": [
      "shN_labels.webp",   // Per-gaussian palette indices (0..count-1)
      "shN_centroids.webp" // Palette of AC coefficients as pixels
    ]
  }
}
```

:::note

* All codebooks contain linear-space values, not sRGB.
* Image data **must** be treated as raw 8-bit integers (no gamma conversion).
* Unless otherwise stated, channels not mentioned are ignored.

:::

---

## 3. Property encodings

### 3.1 Positions

> `means_l.webp`, `means_u.webp` (RGB, 16-bit per axis)

Each axis is quantized to **16 bits** across two images:

```cpp
// 16-bit normalized value per axis (0..65535)
uint16_t qx = (means_u.r << 8) | means_l.r;
uint16_t qy = (means_u.g << 8) | means_l.g;
uint16_t qz = (means_u.b << 8) | means_l.b;

// Dequantize into log-domain nx,ny,nz using per-axis ranges from meta:
float nx = lerp(meta.means.mins[0], meta.means.maxs[0], qx / 65535.0f);
float ny = lerp(meta.means.mins[1], meta.means.maxs[1], qy / 65535.0f);
float nz = lerp(meta.means.mins[2], meta.means.maxs[2], qz / 65535.0f);

// Undo the symmetric log transform used at encode time:
auto unlog = [](float n) -> float {
  float a = std::abs(n);
  float e = std::exp(a) - 1.0f;
  return n < 0.0f ? -e : e;
};

float px = unlog(nx);
float py = unlog(ny);
float pz = unlog(nz);
```

### 3.2 Orientation

> `quats.webp` (RGBA, 26-bit “smallest-three”)

Quaternions are encoded with **3×8-bit components + 2-bit mode** (total **26 bits**) using the standard *smallest-three* scheme.

* **R,G,B** store the three kept (signed) components, uniformly quantized to `[-√2/2, +√2/2]`.
* **A** stores the **mode** in the range **252..255**. The mode is `A - 252` ∈ {0,1,2,3} and identifies which of the four components was the **largest by magnitude** (and therefore omitted from the stream and reconstructed).
* Let `norm = Math.SQRT2` (i.e., √2).

```cpp
// Dequantize the stored three components:
auto toComp = [](uint8_t c) -> float {
  return (c / 255.0f - 0.5f) * 2.0f / std::sqrt(2.0f);
};

float a = toComp(quats.r);
float b = toComp(quats.g);
float c = toComp(quats.b);

uint8_t mode = quats.a - 252; // 0..3 (R,G,B,A is one of the four components)

// Reconstruct the omitted component so that ||q|| = 1 and w.l.o.g. the omitted one is non-negative
float t = a*a + b*b + c*c;
float d = std::sqrt(std::max(0.0f, 1.0f - t));

// Place components according to mode
std::array<float, 4> q;
switch (mode) {
    case 0: q = {d, a, b, c}; break; // omitted = x
    case 1: q = {a, d, b, c}; break; // omitted = y
    case 2: q = {a, b, d, c}; break; // omitted = z
    case 3: q = {a, b, c, d}; break; // omitted = w
    default: throw std::runtime_error("Invalid quaternion mode");
}
```

#### Validity constraints

* `quats.a` **must** be in **252, 253, 254, 255**. Other values are reserved.

### 3.3 Scales

> `scales.webp` (RGB via codebook)

Per-axis sizes are **codebook indices**:

```cpp
float sx = meta.scales.codebook[scales.r]; // 0..255
float sy = meta.scales.codebook[scales.g];
float sz = meta.scales.codebook[scales.b];
```

Interpretation (e.g., principal axis standard deviations vs. full extents) follows the source training setup; values are in **scene units**.

### 3.4 Base color + opacity (DC)

> `sh0.webp` (RGBA)

`sh0` holds the **DC (l=0)** SH coefficient per color channel and **alpha**:

* **R,G,B** are 0..255 indices into `sh0.codebook` (linear domain).
* **A** is the **opacity** in `[0,1]` (i.e., `sh0.a / 255`).

To convert the DC coefficient to **linear RGB** contribution:

```cpp
// SH_C0 = Y_0^0 = 1 / (2 * sqrt(pi))
constexpr float SH_C0 = 0.28209479177387814f;

float r = 0.5f + meta.sh0.codebook[sh0.r] * SH_C0;
float g = 0.5f + meta.sh0.codebook[sh0.g] * SH_C0;
float b = 0.5f + meta.sh0.codebook[sh0.b] * SH_C0;
float a = sh0.a / 255.0f;
```

> **Color space.** Values are **linear**. If you output to sRGB, apply the usual transfer after shading/compositing.

### 3.5 Higher-order SH (optional)

> `shN_labels.webp`, `shN_centroids.webp`

If present, higher-order (AC) SH coefficients are stored via a palette:

* `shN.count` ∈ **\[1,64k]** number of entries.
* `shN.bands` ∈ **\[1,3]** number of bands per entry.

#### Labels

* `shN_labels.webp` stores a **16-bit index** per gaussian with range (0..count-1).

```cpp
uint16_t index = shN_labels.r + (shN_labels.g << 8);
```

#### Centroids (palette)

* `shN_centroids.webp` is an RGB image storing the SH coefficient palette.
* There are always 64 entries per row; entries are packed row-major with origin top-left.

The texture width is dependent on the number of bands:

| Bands | Coefficients | Texure width (pixels) |
|---|---|---|
| 1 | 3 | 64 * 3 = 96 |
| 2 | 8 | 64 * 8 = 512 |
| 3 | 15 | 64 * 15 = 960 |

Calculating the pixel location for spherical harmonic entry n and coefficient c:

```cpp
const std::array<int, 4> coeffs = {0, 3, 8, 15};
int u = (n % 64) * coeffs[bands] + c;
int v = n / 64;
```

---

## 4. Example `meta.json`

```json
{
  "version": 2,
  "count": 187543,
  "means": {
    "mins": [-2.10, -1.75, -2.40],
    "maxs": [ 2.05,  2.25,  1.90],
    "files": ["means_l.webp", "means_u.webp"]
  },
  "scales": {
    "codebook": [/* 256 floats */],
    "files": ["scales.webp"]
  },
  "quats": { "files": ["quats.webp"] },
  "sh0": {
    "codebook": [/* 256 floats */],
    "files": ["sh0.webp"]
  },
  "shN": {
    "count": 128,
    "bands": 3,
    "codebook": [/* 256 floats */],
    "files": ["shN_labels.webp", "shN_centroids.webp"]
  }
}
```

---

## 5. Versioning & compatibility

* Readers **must** check `version`. This document describes **version 2**.
* Additional optional properties may appear in future versions; readers **should** ignore unrecognized fields.

---
