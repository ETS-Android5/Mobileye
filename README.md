# Mobileye - Mobile eye for visually impared

Mobileye is 3rd winner in {M1522.006600}Intelligent System Design Project, CSE, SNU.

This demo application works mono depth estimation in mobile phone with [fast-depth](http://fastdepth.mit.edu/).

It gives vibration notification to user when some hazards like passing cars, motorcycles, even some man with close walk in their real life occurs.

Demo Video:

[![Demovideo](https://img.youtube.com/vi/elPlb0QWnkc/0.jpg)](https://www.youtube.com/watch?v=elPlb0QWnkc)

## Backgrounds

We implement this repo with some backgrounds in [PyTorchDemoApp](https://github.com/pytorch/android-demo-app/tree/master/PyTorchDemoApp).

From main view, with Vision Processing example in main view, you can start some fast-depth options with full convolution or depth-wise separable convolution.

## Fast Start
Install app-debug.apk file with [link](http://cmalab.snu.ac.kr/hb/app-debug.apk)

## Environment
Java17.0.1: [Download](https://www.oracle.com/java/technologies/downloads/)

Android10: Set android studio sdk manager as android10.(Tools - SDK manager)

pytorch_android:1.8.0

pytorch_android_torchvision:1.8.0


## Settings
(Gradle version) In ~/gradle.wrapper/gradle-wrapper.properties
```
distributionUrl=https://services.gradle.org/distributions/gradle-7.2-all.zip
```

(Gradle plugin) In ~/build.gradle
```
dependencies{
	classpath'com.android.tools.build:gradle:7.0.0'
}
```

(Pytorch verision) In app/build.gradle
```
dependencies {
    implementation 'org.pytorch:pytorch_android:1.8.0'
    implementation 'org.pytorch:pytorch_android_torchvision:1.8.0'
}
```

## How to build

Clone or download this repository:

```bash
git clone https://github.com/HanByulKim/Mobileye
```

Clean gradle build:

```bash
./gradlew clean
```

Build gradle with output apk file (path: ~/app/build/outputs/apk/debug/app-debug.apk):

```bash
./gradlew assembleDebug
```

## Citation

If you find our project helpful, please consider to cite our project

```bibTeX
@article{Mobileye2021,
  title = {{{Mobileye}}: {{Visualizing DNN Quantization effect on Network.}},
  shorttitle = {{{Mobileye}}},
  author = {Han-Byul Kim and Seunghun Shin},
  project={2021SNU_ISD},
  year={2021}
}
```