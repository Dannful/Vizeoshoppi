#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include <iostream>

#define BACKSPACE 8
#define ESC 27

using namespace cv;

constexpr char originalWindowTitle[] = "Original";
constexpr char processedWindowTitle[] = "Processed";
constexpr char trackBarName[] = "Intensity";

struct Settings {
    bool exit = false;
    bool grayscale = false;
    bool recording = false;
    bool negative = false;
    unsigned short blur = 0;
    int rotate = 0;
    unsigned short mirrorVertically = 0;
    unsigned short mirrorHorizontally = 0;
    unsigned short canny = 0;
    unsigned short sobel = 0;
    int brightess = 0;
    unsigned short scale = 1;
    unsigned int contrast = 1;
    int trackBarValue = 1;
};

void resetSettings(Settings& settings) {
    memset(&settings, 0, sizeof(Settings) - 3 * sizeof(int));
    settings.contrast = 1;
    settings.scale = 1;
    settings.trackBarValue = 1;
    setTrackbarPos(trackBarName, processedWindowTitle, 1);
}

void processDoUndoEffect(const int key, const char doKey, unsigned short& value) {
    if (key == doKey) {
        if (value == 255)
            return;
        value++;
    }
    if (key == toupper(doKey)) {
        if (value == 0)
            return;
        value--;
    }
}

void processDoUndoEffect(const int key, const char doKey, int& value) {
    if (key == doKey)
        value++;
    if (key == toupper(doKey))
        value--;
}

void processDoUndoEffect(const int key, const char doKey, bool& value) {
    if (key == doKey)
        value = true;
    if (key == toupper(doKey))
        value = false;
}

unsigned short coerceValue(unsigned short value) {
    if (value > 255)
        value = 255;
    if (value < 0)
        value = 0;
    return value;
}

void keyPressed(Settings& settings) {
    const int key = waitKey(1);
    processDoUndoEffect(key, 'b', settings.blur);
    processDoUndoEffect(key, 'g', settings.grayscale);
    processDoUndoEffect(key, 'r', settings.rotate);
    processDoUndoEffect(key, 'c', settings.canny);
    processDoUndoEffect(key, 's', settings.sobel);
    processDoUndoEffect(key, 'k', settings.mirrorHorizontally);
    processDoUndoEffect(key, 'l', settings.mirrorVertically);
    processDoUndoEffect(key, 'n', settings.negative);
    switch (tolower(key)) {
        case 'r':
            settings.recording = false;
            break;
        case '[':
            settings.brightess = settings.trackBarValue;
            break;
        case ']':
            settings.brightess = -settings.trackBarValue;
            break;
        case ';':
            settings.contrast = settings.trackBarValue;
            break;
        case ',':
            settings.recording = false;
            settings.scale <<= 1;
            break;
        case ' ':
            settings.recording = !settings.recording;
            break;
        case BACKSPACE:
            resetSettings(settings);
            break;
        case ESC:
            settings.exit = true;
            break;
        default: break;
    }
}

void processFrame(Mat& result, Settings& settings) {
    if (settings.grayscale)
        cvtColor(result, result, COLOR_RGB2GRAY);
    for (int i = 0; i < settings.blur; i++) {
        if (settings.trackBarValue % 2 == 0)
            settings.trackBarValue++;
        GaussianBlur(result, result, Size(settings.trackBarValue, settings.trackBarValue), 0);
    }
    if (settings.rotate >= 0) {
        for (int i = 0; i < settings.rotate; i++)
            rotate(result, result, ROTATE_90_CLOCKWISE);
    } else {
        const int amount = abs(settings.rotate);
        for (int i = 0; i < amount; i++)
            rotate(result, result, ROTATE_90_COUNTERCLOCKWISE);
    }
    for (int i = 0; i < settings.canny; i++)
        Canny(result.clone(), result, 100, 200);
    for (int i = 0; i < settings.sobel; i++) {
        Mat gradX, gradY;
        Mat absGradX, absGradY;
        Sobel(result, gradX, CV_16S, 1, 0, 1, 1, 0);
        Sobel(result, gradY, CV_16S, 0, 1, 1, 1, 0);
        convertScaleAbs(gradX, absGradX);
        convertScaleAbs(gradY, absGradY);
        addWeighted(absGradX, 0.5, absGradY, 0.5, 0, result);
    }
    for (int i = 0; i < settings.mirrorHorizontally; i++)
        flip(result, result, 1);
    for (int i = 0; i < settings.mirrorVertically; i++)
        flip(result, result, 0);
    if (settings.scale > 1)
        resize(result, result, Size(), 1.0 / settings.scale, 1.0 / settings.scale);
    result.convertTo(result, -1, settings.contrast, settings.brightess);
    if (settings.negative)
        result.convertTo(result, -1, -1, 255);
}

void init(Settings& settings) {
    createTrackbar(trackBarName, processedWindowTitle, nullptr, 255, [](int value, void* data) {
        if (value < 1)
            value = 1;
        *static_cast<int *>(data) = value;
    }, &settings.trackBarValue);
    setTrackbarMin(trackBarName, processedWindowTitle, 1);
    setTrackbarPos(trackBarName, processedWindowTitle, 1);
}

void printHelp() {
    std::cout << "Welcome. Here's the list of commands:" << std::endl;
    std::cout << "b - blur, B - undo last blur" << std::endl;
    std::cout << "g - grayscale, G - undo grayscale" << std::endl;
    std::cout << "r - rotate right, R - rotate left" << std::endl;
    std::cout << "c - Canny, C - undo last Canny" << std::endl;
    std::cout << "s - Sobel, S - undo last Sobel" << std::endl;
    std::cout << "k - flip horizontally" << std::endl;
    std::cout << "l - flip vertically" << std::endl;
    std::cout << "[ - set brightness to positive tracker value" << std::endl;
    std::cout << "] - set brightness to negative tracker value" << std::endl;
    std::cout << "; - set contrast to tracker value" << std::endl;
    std::cout << ", - scale down" << std::endl;
    std::cout << "n - negative, N - undo negative" << std::endl;
    std::cout << "space - record" << std::endl;
    std::cout << "backspace - reset" << std::endl;
    std::cout << "esc - exit" << std::endl;
}

int main(int argc, char** argv) {
    int camera = 0;
    // VideoCapture cap(
    // "C:/Users/vinix/Videos/Star Wars Revenge of the Sith - Obi-Wan VS Anakin Battle. {Full Version HD}.mp4");
    VideoCapture cap(
        "https://192.168.0.32:8080/videofeed");
    // if (!cap.open(camera))
    // return 0;
    Settings settings;
    bool initialized = false;
    VideoWriter video;
    printHelp();
    for (;;) {
        Mat original;
        Mat processed;
        cap >> original;
        cap >> processed;
        if (original.empty()) break;
        keyPressed(settings);
        processFrame(processed, settings);
        if (settings.recording) {
            if (processed.channels() != 3)
                cvtColor(processed, processed, COLOR_GRAY2RGB);
            if (!video.isOpened())
                video.open("output.avi", VideoWriter::fourcc('M', 'J', 'P', 'G'), 60,
                           Size(processed.cols, processed.rows));
            video.write(processed);
        } else {
            if (video.isOpened())
                video.release();
        }
        imshow(originalWindowTitle, original);
        imshow(processedWindowTitle, processed);

        if (settings.exit) break;
        if (!initialized) {
            init(settings);
            initialized = true;
        }
    }
    cap.release(); // release the VideoCapture object
    if (video.isOpened())
        video.release();
    destroyAllWindows();
    return 0;
}
