#include "Segmentacion.h"

Segmentacion::Segmentacion(){

}

Segmentacion::~Segmentacion() {

}

Mat Segmentacion::BalanceBlancos(Mat Imagen) {
    // 1. Convertir a double y normalizar a [0, 1]
    Mat Imagen_double;
    Imagen.convertTo(Imagen_double, CV_64FC3, 1.0 / 255.0);

    // 2. Separar los canales (OpenCV usa BGR por defecto)
    vector<Mat> channels;
    split(Imagen_double, channels);
    Mat B = channels[0];
    Mat G = channels[1];
    Mat R = channels[2];

    // 3. Calcular las medias (mean)
    double mediaR = mean(R)[0];
    double mediaG = mean(G)[0];
    double mediaB = mean(B)[0];

    // 4. Aplicar las correcciones (Evitar división por cero)
    if (mediaR > 0) R = R * (mediaG / mediaR);
    if (mediaB > 0) B = B * (mediaG / mediaB);

    // 5. Concatenar
    vector<Mat> channels_wb = { B, G, R };
    Mat I_wb;
    merge(channels_wb, I_wb);

    // 6. Limitar valores entre 0 y 1 (min/max)
    max(I_wb, 0.0, I_wb);
    min(I_wb, 1.0, I_wb);

    return I_wb;
}

vector<Mat> Segmentacion::AumentoSaturacion(Mat I_wb) {
    // 1. Asegurarnos de que la imagen esté en punto flotante (0.0 a 1.0)
    // Si I_wb ya viene de la función anterior, ya es CV_64F o CV_32F.
    Mat hsvI, img8bit;
    I_wb.convertTo(img8bit, CV_8UC3, 255.0);
    // 2. rgb2hsv(I_wb)
    cvtColor(img8bit, hsvI, COLOR_BGR2HSV);

    // 3. Separar los canales HSV
    vector<Mat> hsv_channels;
    split(hsvI, hsv_channels);

    // hsv_channels[0] es H (Hue)
    // hsv_channels[1] es S (Saturation)
    // hsv_channels[2] es V (Value/Brightness)

    // 4. hsvI(:,:,2) = min(hsvI(:,:,2)*1.55, 1);
    hsv_channels[1] = hsv_channels[1] * 1.55;
    min(hsv_channels[1], 255.0, hsv_channels[1]); // Limitar a 1.0

    

    return hsv_channels;
}

vector<Mat> Segmentacion::CorreccionIluminacion(vector<Mat> hsv_channels) {

    // hsv_channels[0] = H, [1] = S, [2] = V
    Mat V;
    hsv_channels[2].convertTo(V, CV_64F, 1.0 / 255.0);

    // 3. V_bg = imgaussfilt(V, 120);
    // En OpenCV, el sigma es 120. El tamaño del kernel (Size) se puede dejar en (0,0)
    // para que se calcule automáticamente a partir del sigma.
    Mat V_bg;
    GaussianBlur(V, V_bg, Size(0, 0), 120);

    // 4. hsvI(:,:,3) = V ./ max(V_bg, 0.3);
    // Primero aplicamos el máximo para evitar dividir por valores muy pequeños o cero
    Mat V_bg_clamped;
    max(V_bg, 0.3, V_bg_clamped);

    // División elemento a elemento
    Mat V_corregido;
    divide(V, V_bg_clamped, V_corregido);

    V_corregido.convertTo(hsv_channels[2], CV_8U, 255.0);

    return hsv_channels;
}

Mat Segmentacion::SegmentacionImagen(vector<Mat> hsv_channels) {
    Mat hsv, s_norm, mask;

    s_norm = hsv_channels[1]; // Canal Saturación

    // 3. mask = imbinarize(s_norm, graythresh(s_norm));
    // Convertimos a 8 bits temporalmente para usar OTSU (equivalente a graythresh)
    Mat s_8u;
    threshold(s_norm, mask, 0, 255, THRESH_BINARY | THRESH_OTSU);

    // 4. Operaciones Morfológicas (Cierres y Aperturas)
    // strel('disk', n) 
    auto getDisk = [](int n) {
        return getStructuringElement(MORPH_ELLIPSE, Size(2 * n + 1, 2 * n + 1));
        };

    // mask = imclose(mask, strel('disk',4));
    morphologyEx(mask, mask, MORPH_CLOSE, getDisk(4));

    // mask = imopen(mask, strel('disk',6));
    morphologyEx(mask, mask, MORPH_OPEN, getDisk(6));

    // mask = imclose(mask, strel('disk',8));
    morphologyEx(mask, mask, MORPH_CLOSE, getDisk(8));

    // 5. mask = imfill(mask, 'holes');
    // OpenCV no tiene imfill directo, se usa floodFill para llenar huecos
    Mat mask_filled = mask.clone();
    floodFill(mask_filled, Point(0, 0), Scalar(255));
    bitwise_not(mask_filled, mask_filled);
    mask = (mask | mask_filled);

    // 6. mask = imclearborder(mask);
    // Eliminar objetos que tocan el borde
    Rect rect(0, 0, mask.cols, mask.rows);
    rectangle(mask, rect, Scalar(0), 2); // Dibujar borde negro de 2px

    return mask;
}

Mat Segmentacion::FiltrarObjetoLego(Mat mask) {
    // 1. CC = bwconncomp(mask) y regionprops
    // En OpenCV usamos findContours para obtener los objetos
    vector<vector<Point>> contours;
    findContours(mask, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

    if (contours.empty()) return Mat::zeros(mask.size(), CV_8UC1);

    // 2. Calcular áreas y encontrar la máxima
    vector<double> areas;
    double maxArea = 0;
    for (const auto& contour : contours) {
        double a = contourArea(contour);
        areas.push_back(a);
        if (a > maxArea) maxArea = a;
    }

    // 3. Crear una máscara vacía (BW = false(size(mask)))
    Mat BW = Mat::zeros(mask.size(), CV_8UC1);

    // 4. Filtrar: props.Area >= 0.1 * max(props.Area)
    double thresholdArea = 0.1 * maxArea;

    for (int i = 0; i < contours.size(); i++) {
        if (areas[i] >= thresholdArea) {
            // Dibujar el contorno relleno (equivale a asignar PixelIdxList)
            // -1 significa rellenar el interior del contorno
            drawContours(BW, contours, i, Scalar(255), -1);
        }
    }

    return BW;
}

Segmentacion::ObjetosSegmentados Segmentacion::RecorteAjusteImagen(vector<Mat> hsv_channels, Mat BW) {

    ObjetosSegmentados resultado;
    Size tam_final(200, 200);
    int extra = 20;

	Mat I_hsv,I_original;
	merge(hsv_channels, I_hsv);
    cvtColor(I_hsv, I_original, COLOR_HSV2BGR);

    // SE = strel('square', 15);
    Mat SE = getStructuringElement(MORPH_RECT, Size(15, 15));

    // Encontrar componentes (props en MATLAB)
    vector<vector<Point>> contours;
    findContours(BW, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

    for (size_t k = 0; k < contours.size(); k++) {
        // 1. Obtener BoundingBox (bb)
        Rect bb = boundingRect(contours[k]);

        // 2. Aplicar el "extra" (Padding) con límites de imagen
        int x = max(0, bb.x - extra);
        int y = max(0, bb.y - extra);
        int w = min(I_original.cols - x, bb.width + 2 * extra);
        int h = min(I_original.rows - y, bb.height + 2 * extra);

        Rect roi(x, y, w, h);

        // 3. Crear mascara local para este objeto específico
        Mat mask_local = Mat::zeros(BW.size(), CV_8UC1);
        drawContours(mask_local, contours, static_cast<int>(k), Scalar(255), -1);

        // 4. Recortar (imcrop)
        Mat mask_crop = mask_local(roi).clone();
        Mat I_crop = I_original(roi).clone();

        // 5. Limpieza morfológica local
        morphologyEx(mask_crop, mask_crop, MORPH_OPEN, SE);
        morphologyEx(mask_crop, mask_crop, MORPH_CLOSE, SE);

        // 6. Aplicar máscara a la imagen (I_crop_masked)
        // Como I_original es double [0,1], mask_crop debe convertirse a ese tipo
        Mat mask_float;
        mask_crop.convertTo(mask_float, I_original.type(), 1.0 / 255.0);

        Mat I_crop_masked;
        // Multiplicación canal por canal
        vector<Mat> channels;
        split(I_crop, channels);
        for (int i = 0; i < 3; i++) {
            multiply(channels[i], mask_float, channels[i]);
        }
        merge(channels, I_crop_masked);

        // 7. Redimensionar (imresize)
        Mat I_norm, BW_norm;
        resize(I_crop_masked, I_norm, tam_final);
        // 'nearest' para la máscara binaria para no crear grises en los bordes
        resize(mask_crop, BW_norm, tam_final, 0, 0, INTER_NEAREST);

        resultado.imagenesColor.push_back(I_norm);
        resultado.mascarasBin.push_back(BW_norm);
    }

    return resultado;
}