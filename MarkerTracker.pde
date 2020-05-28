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

    // A function to check if a contour is a good candidate
    boolean checkContourCondition(MatOfPoint2f approxCurve, Mat image_bgr, int kNumOfCorners) {
        // Get a bounding rectangle of the approximated contour
        Rect bounding_rectangle = Imgproc.boundingRect(new MatOfPoint(approxCurve.toArray()));
        double marker_size = bounding_rectangle.area();

        // Filter bad contours
        int kImageSize = image_bgr.rows() * image_bgr.cols();
        double kMarkerSizeMin = kImageSize * 0.01;
        double kMarkerSizeMax = kImageSize * 0.99;
        boolean is_contour_valid = (marker_size > kMarkerSizeMin) 
            && (marker_size < kMarkerSizeMax)
            && approxCurve.size().height == kNumOfCorners
            && Imgproc.isContourConvex(new MatOfPoint(approxCurve.toArray()));

        return is_contour_valid;
    }

    // obtain the value of sampled points (from gray_img) in a stripe
    int subpixSampleSafe(Mat pSrc, PVector p) {
        int x = (int)(floor(p.x));
        int y = (int)(floor(p.y));

        if (x < 0 || x >= pSrc.cols() - 1 || y < 0 || y >= pSrc.rows() - 1)
            return 127;

        int dx = (int)(256 * (p.x - floor(p.x)));
        int dy = (int)(256 * (p.y - floor(p.y)));

        int i   = (int)(pSrc.get(y,   x  )[0]);
        int ix  = (int)(pSrc.get(y,   x+1)[0]);
        int iy  = (int)(pSrc.get(y+1, x  )[0]);
        int ixy = (int)(pSrc.get(y+1, x+1)[0]);

        int a = i  + ((dx * (ix  - i )) >> 8);
        int b = iy + ((dx * (ixy - iy)) >> 8);

        return a + ((dy * (b - a)) >> 8);
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

        // PImage dst = createImage(image_width, image_height, 1);
        // opencv.toPImage(image_gray_filtered, dst);
        // image(dst, 0, 0);


        // add for the first exercise - extract contours and find points along the rough contours
        // **************************************************************

        // contour detection (exercise part)
        image_gray_filtered = OpenCV.imitate(opencv.getGray()); //??add by myself
        Imgproc.threshold(image_gray, image_gray_filtered, thresh, 255, Imgproc.THRESH_BINARY);

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

        // for each contour found
        for (int i = 0; i < contours.size(); i++) {
            //Convert contours from MatOfPoint to MatOfPoint2f
            MatOfPoint2f contour2f = new MatOfPoint2f(contours.get(i).toArray());
            //Processing on mMOP2f1 which is in type MatOfPoint2f
            double approxDistance = Imgproc.arcLength(contour2f, true) * 0.02; //kEpsilon in the commented code
            //double kEpsilon = 0.02 * Imgproc.arcLength(new MatOfPoint2f(contour.toArray()), true);

            if (approxDistance > 1) {
                //Find Polygons
                Imgproc.approxPolyDP(contour2f, approxCurve, approxDistance, true);

                // this part is self code, but will try to use the provided code
                // **************************************************************

                // //Convert back to MatOfPoint
                // MatOfPoint points = new MatOfPoint(approxCurve.toArray());

                // Scalar dot_color = new Scalar(0, 255, 0); //0.8 is transparency

                // //Rectangle Checks - Points, area, convexity
                // if (points.total() == kNumOfCorners && Math.abs(Imgproc.contourArea(points)) > kMarkerSizeLength && Imgproc.isContourConvex(points)){
                //     // print(i);
                //     // println(points.size());
                //     Point[] points_array = points.toArray();
                //     // println(x.length); //4
                //     for (int j = 0; j < points_array.length; j++){
                //         float x = (float) points_array[j].x;
                //         float y = (float) points_array[j].y;
                //         circle(x, y, 10); //x(j) = (int x, int y) // this circle() is Processing function
                //         println("here");
                //         for (int k = j+1; k < points_array.length; k++){
                //             if ((k - j == 1) || (k - j == 3)){
                //                 float x_2 = (float) points_array[k].x;
                //                 float y_2 = (float) points_array[k].y;
                //                 line(x, y, x_2, y_2); //line(x1, y1, x2, y2)
                //                 for (int l = 1; l < 7; l++){
                //                     circle(((7-l)*x + l*x_2)/7, ((7-l)*y + l*y_2)/7, 5);
                //                 }
                //             }
                //         }
                //     }
                // }
                // **************************************************************
                
                //Convert back to MatOfPoint
                MatOfPoint points = new MatOfPoint(approxCurve.toArray());
                // put as Rect class
                Rect kRectangle = Imgproc.boundingRect(points);
                double kMarkerSize = kRectangle.area();

                //check the condition of detected contours, function written above
                boolean is_valid = checkContourCondition(approxCurve, image_bgr, kNumOfCorners);

                if (!is_valid)
                    continue;

                if (MARKER_TRACKER_DEBUG) {
                    // Draw lines
                    noFill();
                    strokeWeight(4);
                    stroke(random(255), random(255), random(255));

                    beginShape();
                    Point[] p = approxCurve.toArray();
                    for (int j = 0; j < p.length; j++) {
                        vertex((float)p[j].x, (float)p[j].y);
                    }
                    endShape(CLOSE);
                }

                Mat[] line_parameters = new Mat[4];

                // loop for each edge
                for (int k = 0; k < kNumOfCorners; k++) {
                    int kNumOfEdgePoints = 7;

                    if (MARKER_TRACKER_DEBUG) {
                        // Draw corner points
                        float kCircleSize = 10;
                        fill(0, 255, 0);
                        noStroke();
                        Point[] p = approxCurve.toArray();
                        circle((float)p[k].x, (float)p[k].y, kCircleSize);
                    }

                    Point[] approx_points = approxCurve.toArray();
                    PVector pa = OpenCV.pointToPVector(approx_points[(k+1) % kNumOfCorners]);
                    PVector pb = OpenCV.pointToPVector(approx_points[k]);
                    PVector kEdgeDirectionVec = PVector.div(PVector.sub(pa, pb), kNumOfEdgePoints);
                    float kEdgeDirectionVecNorm = kEdgeDirectionVec.mag();

                    int stripe_length = (int)(0.8 * kEdgeDirectionVecNorm);
                    if (stripe_length < 5)
                        stripe_length = 5;

                    stripe_length |= 1;

                    int kStop = stripe_length >> 1;
                    int kStart = -kStop;

                    int kStripeWidth = 3;
                    Size kStripeSize = new Size(kStripeWidth, stripe_length);

                    PVector kStripeVecX = PVector.div(kEdgeDirectionVec, kEdgeDirectionVecNorm);
                    PVector kStripeVecY = new PVector(-kStripeVecX.y, kStripeVecX.x);

                    Mat stripe_image = new Mat(kStripeSize, CvType.CV_8UC1);

                    // Array for edge point centers
                    PVector[] edge_points = new PVector[kNumOfEdgePoints - 1];
                    
                    // loop for each point in an edge
                    for (int j = 1; j < kNumOfEdgePoints; j++) {
                        PVector edge_point = PVector.add(pb, PVector.mult(kEdgeDirectionVec, j));

                        if (MARKER_TRACKER_DEBUG) {
                            // Draw line delimeters
                            fill(0, 0, 255);
                            noStroke();
                            circle(edge_point.x, edge_point.y, 4);
                        }
                        // exercise 2 - find exact edge points
                        // **************************************************************
                        // draw strips
                        for (int m = -1; m <= 1; m++) {
                            for (int n = kStart; n <= kStop; n++) {
                                PVector subpixel = PVector.add(
                                    PVector.add(edge_point, PVector.mult(kStripeVecX, m)), // strip width only 3 (-1,0,1)
                                    PVector.mult(kStripeVecY, n) //strip lenght (n, -kstart to kstart)
                                );

                                if (MARKER_TRACKER_DEBUG) {
                                    noStroke();
                                    if (isFirstStripe) {
                                        fill(255, 0, 255);
                                        circle(subpixel.x, subpixel.y, 2);
                                    } else {
                                        fill(0, 255, 255);
                                        circle(subpixel.x, subpixel.y, 2);
                                    }
                                }

                                // Fetch subpixel value
                                int kSubpixelValue = subpixSampleSafe(image_gray, subpixel);
                                int kStripeX = m + 1; // update kStripeX for next dot
                                int kStripeY = n + (stripe_length >> 1); // update kStripeY for next dot
                                stripe_image.put(kStripeY, kStripeX, kSubpixelValue); // store in matrix stripe_image
                            }
                        }

                        // use sobel operator on stripe
                        // ( -1 , -2, -1 )
                        // (  0 ,  0,  0 )
                        // (  1 ,  2,  1 )

                        double[] sobelValues = new double[stripe_length - 2];
                    
                        for (int n = 1; n < stripe_length - 1; n++) {
                            byte[] p = new byte[3];

                            stripe_image.get(n - 1, 0, p);
                            double r1 = -(p[0] & 0xFF) - 2.0 * (p[1] & 0xFF) - (p[2] & 0xFF); // 0xFF is masking

                            stripe_image.get(n + 1, 0, p);
                            double r3 =  (p[0] & 0xFF) + 2.0 * (p[1] & 0xFF) + (p[2] & 0xFF);

                            sobelValues[n - 1] = -(r1 + r3);
                        }

                        double maxVal = -1;
                        int maxIndex = 0;
                        for (int n = 0; n < stripe_length - 2; n++) {
                            if (sobelValues[n] > maxVal) {
                                maxVal = sobelValues[n];
                                maxIndex = n;
                            }
                        }

                        double y0, y1, y2;
                        y0 = (maxIndex <= 0) ? 0 : sobelValues[maxIndex - 1];
                        y1 = sobelValues[maxIndex];
                        y2 = (maxIndex >= stripe_length - 3) ? 0 : sobelValues[maxIndex + 1];

                        // formula for calculating the x-coordinate of the vertex of a parabola,
                        // given 3 points with equal distances
                        // (xv means the x value of the vertex, d the distance between the points):
                        // xv = x1 + (d / 2) * (y2 - y0)/(2*y1 - y0 - y2)
                        double pos = (y2 - y0) / (4 * y1 - 2 * y0 - 2 * y2);

                        // exact point with subpixel accuracy
                        int maxIndexShift = maxIndex - (stripe_length >> 1); // ???
                        PVector edgeCenter = PVector.add(edge_point, PVector.mult(kStripeVecY, maxIndexShift + (float)pos));

                        if (MARKER_TRACKER_DEBUG) {
                            fill(0, 0, 255);
                            noStroke();
                            circle(edgeCenter.x, edgeCenter.y, 4);
                        }
                        edge_points[j - 1] = new PVector(edgeCenter.x, edgeCenter.y);

                        if (isFirstStripe) {
                            if (MARKER_TRACKER_DEBUG) {
                                // TODO: move stripe_image to another window
                                PImage dst_stripe = createImage(100, 300, ARGB);
                                Mat iplTmp = new Mat(new Size(100, 300), CvType.CV_8UC1);
                                Imgproc.resize(stripe_image, iplTmp, new Size(100, 300), 0.0, 0.0, Imgproc.INTER_NEAREST);
                                opencv.toPImage(iplTmp, dst_stripe);
                                image(dst_stripe, 0, 0);
                            }
                            isFirstStripe = false;
                        }
                        // **************************************************************

                    } // --- end of loop over edge points of one edge
                }
            } // end of loop over contour candidates
        }
        // **************************************************************
    }
}