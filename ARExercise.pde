import gab.opencv.*;
import processing.video.*;

final boolean MARKER_TRACKER_DEBUG = true;

final boolean USE_SAMPLE_IMAGE = true;

// We've found that some Windows build-in cameras (e.g. Microsoft Surface)
// cannot work with processing.video.Capture.*.
// Instead we use DirectShow Library to launch these cameras.
final boolean USE_DIRECTSHOW = true;

final double kMarkerSize = 1000; // marker area

Capture cap;
DCapture dcap;
OpenCV opencv;

ArrayList<Marker> markers;
MarkerTracker markerTracker;

PImage img;

void selectCamera() {
  String[] cameras = Capture.list();

  if (cameras == null) {
    println("Failed to retrieve the list of available cameras, will try the default");
    cap = new Capture(this, 640, 480);
  } else if (cameras.length == 0) {
    println("There are no cameras available for capture.");
    exit();
  } else {
    println("Available cameras:");
    printArray(cameras);

    // The camera can be initialized directly using an element
    // from the array returned by list():
    //cap = new Capture(this, cameras[5]);

    // Or, the settings can be defined based on the text in the list
    cap = new Capture(this, 1280, 720, "USB2.0 HD UVC WebCam", 10);
  }
}


void settings() {
  if (USE_SAMPLE_IMAGE) {
    size(1000, 730);
    opencv = new OpenCV(this, "./marker_test.jpg");
  } else {
    if (USE_DIRECTSHOW) {
      dcap = new DCapture();
      size(dcap.width, dcap.height);
      opencv = new OpenCV(this, dcap.width, dcap.height);
    } else {
      selectCamera();
      size(cap.width, cap.height);
      opencv = new OpenCV(this, cap.width, cap.height);
    }
  }
}

void setup() {
  smooth();
  markerTracker = new MarkerTracker(kMarkerSize);

  if (!USE_DIRECTSHOW) {
    cap.start();
  }
}

void draw() {
  if (!USE_SAMPLE_IMAGE) {
    if (USE_DIRECTSHOW) {
      img = dcap.updateImage();
      opencv.loadImage(img);
    } else {
      if (cap.width <= 0 || cap.height <= 0) {
        println("Incorrect capture data. continue");
        return;
      }
      opencv.loadImage(cap);
    }
  }

  markerTracker.findMarker(markers);
  System.gc(); //gabage collecting
}

void captureEvent(Capture c) {
  if (!USE_DIRECTSHOW && c.available())
      c.read();
}
