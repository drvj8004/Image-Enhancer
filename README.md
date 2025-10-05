# Face Enhancer

A small C++ program that makes soft, realistic improvements to faces in photos. It upscales the face region with a deep-learning super-resolution model, cleans noise, lifts local contrast, sharpens gently, then blends the result back into the original image. If no face is found, you can process the whole image instead.

---

## What the program does

- Finds the largest face in the photo (DNN face detector when available, optional Haar cascade fallback)
- Expands the box to include hair and surrounding context and clamps it inside bounds
- Upscales the face with a super‑resolution model (EDSR ×4 by default)
- Runs a **natural** enhancement chain:
  - gamma correction (exposure)
  - edge‑preserving denoise (bilateral)
  - micro‑contrast via `detailEnhance`
  - CLAHE on luminance
  - gentle unsharp mask
- Resizes the enhanced face back to the crop size and merges it with **seamlessClone** and a feather mask
- Optional gentle global pass (low‑clip CLAHE and tiny unsharp)

Defaults aim for a balanced, realistic look — not crunchy contrast.

---

## What it uses

- **OpenCV 4.x** (with **opencv_contrib**) for:
  - `dnn_superres` (super‑resolution)
  - `photo` (detail enhancement, seamless clone)
  - `dnn` (face detector)
- **Models**
  - Super‑resolution: `EDSR_x4.pb` (default)
  - Face detector (preferred): `opencv_face_detector.prototxt` + `opencv_face_detector.caffemodel`
  - Optional Haar fallback: `haarcascade_frontalface_default.xml`

---

## Suggested folder layout

```
project/
├─ face_enhancer.cpp
├─ models/
│  ├─ EDSR_x4.pb
│  ├─ opencv_face_detector.prototxt
│  ├─ opencv_face_detector.caffemodel
│  └─ haarcascade_frontalface_default.xml   (optional)
├─ input.jpg
└─ output.png
```

---

## Build requirements

- C++17 compiler
- OpenCV 4.1+ with contrib modules
- On Windows, either prebuilt OpenCV with contrib, or vcpkg

---

## Build — macOS (Terminal)

1) Install OpenCV
```bash
brew install opencv
```

2) Compile
```bash
clang++ face_enhancer.cpp -std=c++17 -O3   $(pkg-config --cflags --libs opencv4)   -o face_enhancer
```

If `pkg-config` is not available, point to OpenCV manually, for example:
```bash
clang++ face_enhancer.cpp -std=c++17 -O3   -I/usr/local/include/opencv4   -L/usr/local/lib   -lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_photo   -lopencv_dnn -lopencv_dnn_superres -lopencv_xphoto -lopencv_imgcodecs   -o face_enhancer
```

---

## Build — Windows (MSVC) with CMake + vcpkg

1) Install vcpkg and then:
```powershell
vcpkg install opencv[contrib]:x64-windows
```

2) Create **CMakeLists.txt** next to `face_enhancer.cpp`:
```cmake
cmake_minimum_required(VERSION 3.20)
project(face_enhancer CXX)
set(CMAKE_CXX_STANDARD 17)

find_package(OpenCV REQUIRED)
add_executable(face_enhancer face_enhancer.cpp)
target_link_libraries(face_enhancer PRIVATE ${OpenCV_LIBS})
```

3) Configure and build:
```powershell
cmake -S . -B build -DCMAKE_TOOLCHAIN_FILE=C:\path\to\vcpkg.cmake -DVCPKG_TARGET_TRIPLET=x64-windows
cmake --build build --config Release
```

The executable will be in `build\Release\face_enhancer.exe`.

(If you have a prebuilt OpenCV, set `OpenCV_DIR` to your OpenCV build/install folder and run CMake without the vcpkg toolchain.)

---

## Build — VS Code

- Install the **CMake Tools** extension.
- Put the `CMakeLists.txt` shown above in your project root.
- Open the folder in VS Code, pick a compiler kit, click **Configure**, then **Build**.
- Set program arguments in **Run and Debug**, or run from the integrated terminal.

---

## Run

By default the program looks in `models/` for the networks. You can override paths with flags.

**DNN face detector + EDSR ×4**
```bash
# macOS/Linux
./face_enhancer input.jpg output.png   --sr models/EDSR_x4.pb --scale 4   --proto models/opencv_face_detector.prototxt   --weights models/opencv_face_detector.caffemodel
```
```powershell
# Windows
.uild\Releaseace_enhancer.exe input.jpg output.png `
  --sr models/EDSR_x4.pb --scale 4 `
  --proto models\opencv_face_detector.prototxt `
  --weights models\opencv_face_detector.caffemodel
```

**No detector available — process whole frame**
```bash
./face_enhancer input.jpg output.png --no-face-only --sr models/EDSR_x4.pb
```

**Use Haar fallback when DNN files are not present**
```bash
./face_enhancer input.jpg output.png   --cascade models/haarcascade_frontalface_default.xml
```

---

## Flags (natural look with gentle controls)

- `--sr <pb>` super‑resolution model, default `models/EDSR_x4.pb`
- `--scale 2|3|4|8` upscale factor, default 4
- `--proto <file>` DNN face prototxt
- `--weights <file>` DNN face caffemodel
- `--cascade <xml>` Haar cascade fallback
- `--no-face-only` process the whole image instead of just the face region
- `--no-final` skip the gentle global finishing pass

**Fine‑tune the natural look**
- `--clip <float>` local contrast (CLAHE on luminance), default `1.2`
- `--gclip <float>` global CLAHE after blending, default `0.0` (off)
- `--sharp <float>` unsharp amount on the face patch, default `0.35`
- `--gsharp <float>` global unsharp amount, default `0.15`
- `--gamma <float>` exposure, default `1.0` (e.g., `1.05` slightly brighter)

Examples:
```bash
# Softer output
./face_enhancer input.jpg output.png --clip 1.0 --sharp 0.25 --no-final

# A touch more pop
./face_enhancer input.jpg output.png --clip 1.4 --sharp 0.45 --gsharp 0.15
```

---






