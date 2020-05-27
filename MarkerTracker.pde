import gab.opencv.*;
import org.opencv.imgproc.Imgproc;

import org.opencv.core.Core;

import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.CvType;

import org.opencv.core.Point;
import org.opencv.core.Size;
import org.opencv.core.Rect;

import org.opencv.core.Scalar;

class Marker {
    int code;
    float[] pose;

    Marker() {
        pose = new float[16];
    }

    void print_matrix() {
        int kSize = 4;
        for (int r = 0; r < kSize; r++) {
            for (int c = 0; c < kSize; c++) {
                println(pose[r + kSize * (c - 1)]);
            }
        }
    }
}

class MarkerTracker {
    int thresh;     // Threshold: gray to mono
    int bw_thresh;  // Threshold for gray marker to ID image
    double kMarkerSizeLength; //area not lenght
    int kNumOfCorners;

    Mat image_bgr;
	Mat image_gray;
	Mat image_gray_filtered;

    MarkerTracker(double _kMarkerSizeLength) {
        thresh = 80;
        bw_thresh = 100;
        kMarkerSizeLength = _kMarkerSizeLength;
        init();
    }

    MarkerTracker(double _kMarkerSizeLength, int _thresh, int _bw_thresh) {
        thresh = _thresh;
        bw_thresh = _bw_thresh;
        kMarkerSizeLength = _kMarkerSizeLength;
        init();
    }

    void init() {
        println("Startup");
    }

    void cleanup() {
        println("Finished");
    }

	void findMarker(ArrayList<Marker> markers) {
        boolean isFirstStripe = true;
        boolean isFirstMarker = true;

        //exercise part
        ArrayList<MatOfPoint> contours = new ArrayList<MatOfPoint>();
        Mat hierarchy = new Mat(); //not important in our case, for contour in contour..

        image_bgr = OpenCV.imitate(opencv.getColor());
        opencv.getColor().copyTo(image_bgr);
        int image_height = image_bgr.rows();
        int image_width  = image_bgr.cols();

        PImage dst = createImage(image_width, image_height, 1);
        opencv.toPImage(image_bgr, dst);
        image(dst, 0, 0);

        image_gray = OpenCV.imitate(opencv.getGray());
        opencv.getGray().copyTo(image_gray);

        // PImage dst = createImage(image_width, image_height, ARGB);
        // opencv.toPImage(image_gray, dst);
        // image(dst, 0, 0);

        // contour detection (exercise part)
        image_gray_filtered = OpenCV.imitate(opencv.getGray()); //??add by myself
        Imgproc.threshold(image_gray, image_gray_filtered, thresh, 255, Imgproc.THRESH_BINARY);

        // PImage dst = createImage(image_width, image_height, 1);
        // opencv.toPImage(image_gray_filtered, dst);
        // image(dst, 0, 0);

        // find contours
        Imgproc.findContours(image_gray_filtered, contours, hierarchy, Imgproc.RETR_LIST, Imgproc.CHAIN_APPROX_SIMPLE);
        
        //test
        // Imgproc.drawContours(image_bgr, contours, i, dot_color, 2);
        // PImage dst = createImage(image_width, image_height, 1);
        // opencv.toPImage(image_bgr, dst);
        // image(dst, 0, 0);

        //https://www.programcreek.com/java-api-examples/?class=org.opencv.imgproc.Imgproc&method=drawContours
        //For conversion later on
        MatOfPoint2f approxCurve = new MatOfPoint2f();
        kNumOfCorners = 4;
        Scalar dot_color = new Scalar(0, 255, 0); //0.8 is transparency

        // for each contour found
        for (int i = 0; i < contours.size(); i++) {
            //Convert contours from MatOfPoint to MatOfPoint2f
            MatOfPoint2f contour2f = new MatOfPoint2f(contours.get(i).toArray());
            //Processing on mMOP2f1 which is in type MatOfPoint2f
            double approxDistance = Imgproc.arcLength(contour2f, true) * 0.02; //epsilon in the commented code

            if (approxDistance > 1) {
                //Find Polygons
                Imgproc.approxPolyDP(contour2f, approxCurve, approxDistance, true);
                //Convert back to MatOfPoint
                MatOfPoint points = new MatOfPoint(approxCurve.toArray());
                //Rectangle Checks - Points, area, convexity
                if (points.total() == kNumOfCorners && Math.abs(Imgproc.contourArea(points)) > kMarkerSizeLength && Imgproc.isContourConvex(points)){
                    // print(i);
                    // println(points.size());
                    Point[] points_array = points.toArray();
                    // println(x.length); //4
                    for (int j = 0; j < points_array.length; j++){
                        float x = (float) points_array[j].x;
                        float y = (float) points_array[j].y;
                        circle(x, y, 10); //x(j) = (int x, int y) // this circle() is Processing function
                        println("here");
                        for (int k = j+1; k < points_array.length; k++){
                            if ((k - j == 1) || (k - j == 3)){
                                float x_2 = (float) points_array[k].x;
                                float y_2 = (float) points_array[k].y;
                                line(x, y, x_2, y_2); //line(x1, y1, x2, y2)
                                for (int l = 1; l < 7; l++){
                                    circle(((7-l)*x + l*x_2)/7, ((7-l)*y + l*y_2)/7, 5);
                                }
                            }
                        }
                    }
                }

            }
        }
        // for (MatOfPoint contour: contours) {
            
        //     MatOfPoint2f contour_approx = new MatOfPoint2f();
        //     double kEpsilon = 0.05 * Imgproc.arcLength(new MatOfPoint2f(contour.toArray()),true);

        //     Imgproc.approxPolyDP(new MatOfPoint2f(contour.toArray()), contour_approx,kEpsilon,true);

        //     //check size
        //     Rect bounding_rectangle = Imgproc.boundingRect(new MatOfPoint(contour_approx.toArray()));
        //     double marker_size = bounding_rectangle.area();
        //     boolean is_contour_valid = (marker_size > kMarkerSizeLength)&&
        //     // (marker_size < kMarkerSizeMax)&&
        //     (contour_approx.size().height == kNumOfCorners)&&
        //     (Imgproc.isContourConvex(new MatOfPoint(contour_approx.toArray())));

        //     print(bounding_rectangle);
        //     println(is_contour_valid);
        //     if(is_contour_valid = false) {  //<-- continue leaks.. why?
        //         continue;
        //     }
            // Point[] p = contour_approx.toArray();
            // for (int i = 0; i < p.length; i++){
            //     println(i);
            //     circle(4,5,20);
            //     // circle((float)p[i].x, (float)p[i].y), (float)5);
            // }
            // ...
            
        // }
    }
}