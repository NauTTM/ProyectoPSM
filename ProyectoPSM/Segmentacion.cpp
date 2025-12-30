#include "Segmentacion.h"

Segmentacion::Segmentacion() {
}

Segmentacion::~Segmentacion() {

}


void Segmentacion::SegmentarImagen(const Mat& Imagen) {
    //Mat img = imread("dataset/01_000_10_001.jpg");
        Mat sat, img1, img8bit;
		vector<vector<Point>> img2;
        Mat blanco = BalanceBlancos(Imagen);
        //blanco.convertTo(img1, CV_8UC3, 255.0);
        vector<Mat> vecSat = AumentoSaturacion(blanco);
        /*merge(vecSat, sat);
        cvtColor(sat, img8bit, COLOR_HSV2BGR);
        img8bit.convertTo(img1, CV_8UC3, 255.0);*/
        vector<Mat> corr = CorreccionIluminacion(vecSat);
        //merge(corr, sat);
        //cvtColor(sat, img8bit, COLOR_HSV2BGR);
        //img8bit.convertTo(img1, CV_8UC3, 255.0);
       Mat mask = SegmentacionImagen(corr);
       //mask.convertTo(img1, CV_8UC1, 255.0);
     // img1 = mask;
       /* int type = mask.type();*/
        Mat bw = FiltrarObjetoLego(mask);
       // bw.convertTo(img1, CV_8UC1, 255.0);
        Segmentacion::ObjetosSegmentados objetos = RecorteAjusteImagen(corr, bw);
       img1 = objetos.imagenesColor[0];
        img2 = MostrarBordes(bw);
        emit SegmentacionCompletada(img1, img2);
        
}

Mat Segmentacion::BalanceBlancos(const Mat& Imagen) {
    //// 1. Convertir a float y normalizar a [0, 1]
    Mat I_wb;
    Imagen.convertTo(I_wb, CV_32FC3, 1.0 / 255.0);

    //// 2. Separar los canales (OpenCV usa BGR por defecto)
    //vector<Mat> channels;
    //split(Imagen_double, channels);
    //Mat B = channels[0];
    //Mat G = channels[1];
    //Mat R = channels[2];

	Scalar meanVal = mean(I_wb);
    //// 3. Calcular las medias (mean)
    float mediaB = meanVal[0];
    float mediaG = meanVal[1];
    float mediaR = meanVal[2];

    //// 4. Aplicar las correcciones (Evitar división por cero)
    float kR = (mediaR > 0.f) ? (mediaG / mediaR) : 1.f;
    float kB = (mediaB > 0) ? (mediaG / mediaB) : 1.f;

    //// 5. Concatenar
    //vector<Mat> channels_wb = { B, G, R };
    //Mat I_wb;
    //merge(channels_wb, I_wb);

	vector<Mat> canales;
	split(I_wb, canales);
	canales[2] = canales[2] * kR; // Canal R
	canales[0] = canales[0] * kB; // Canal B
	merge(canales, I_wb);

    //// 6. Limitar valores entre 0 y 1 (min/max)
    //max(I_wb, 0.0, I_wb);
    //min(I_wb, 1.0, I_wb);


    return I_wb;
}

vector<Mat> Segmentacion::AumentoSaturacion(const Mat& I_wb) {
    // 1. Asegurarnos de que la imagen esté en punto flotante (0.0 a 1.0)
    // Si I_wb ya viene de la función anterior, ya es CV_64F o CV_32F.
    Mat hsvI;
    //I_wb.convertTo(img8bit, CV_8UC3, 255.0);
    // 2. rgb2hsv(I_wb)
    cvtColor(I_wb, hsvI, COLOR_BGR2HSV);

    // 3. Separar los canales HSV
    vector<Mat> hsv_channels;
    split(hsvI, hsv_channels);

    // hsv_channels[0] es H (Hue)
    // hsv_channels[1] es S (Saturation)
    // hsv_channels[2] es V (Value/Brightness)

    // 4. hsvI(:,:,2) = min(hsvI(:,:,2)*1.55, 1);
    hsv_channels[1] *= 1.55f;
    min(hsv_channels[1], 1.0f, hsv_channels[1]); // Limitar a 1.0

    

    return hsv_channels;
}

vector<Mat> Segmentacion::CorreccionIluminacion(const vector<Mat> &hsv_channels) {


    // hsv_channels[0] = H, [1] = S, [2] = V
    Mat V, V_bg, V_small, V_bg_small;
    //hsv_channels[2].convertTo(V, CV_64F, 1.0 / 255.0);
	V = hsv_channels[2];
    // 1. Reducir solo para calcular el fondo (1/4 del tamaño)
    resize(V, V_small, Size(), 0.25, 0.25, INTER_LINEAR);
    // 2. El sigma ahora es proporcional (120 / 4 = 30)
    GaussianBlur(V_small, V_bg_small, Size(0, 0), 30.0);
    // 3. Regresar al tamaño original
    resize(V_bg_small, V_bg, V.size(), 0, 0, INTER_LINEAR);

    // 4. hsvI(:,:,3) = V ./ max(V_bg, 0.3);
    // Primero aplicamos el máximo para evitar dividir por valores muy pequeños o cero
    max(V_bg, 0.3, V_bg);

    // División elemento a elemento
    Mat V_corregido;
    divide(V, V_bg, V_corregido);

	min(V_corregido, 1.0, V_corregido); // Limitar a 1.0
	vector<Mat> hsv_channels_corregidos = hsv_channels;
	hsv_channels_corregidos[2] = V_corregido;
    //V_corregido.convertTo(hsv_channels[2], CV_8U, 255.0);

    return hsv_channels_corregidos;
}

Mat Segmentacion::SegmentacionImagen(const vector<Mat> &hsv_channels) {
    Mat hsv, s_norm, mask;

    hsv_channels[1].convertTo(s_norm, CV_8UC1, 255.0); // Canal Saturación

    // 3. mask = imbinarize(s_norm, graythresh(s_norm));
    // Convertimos a 8 bits temporalmente para usar OTSU (equivalente a graythresh)
    Mat s_8u;
    threshold(s_norm, mask, 0, 255, THRESH_BINARY | THRESH_OTSU);

    // 4. Operaciones Morfológicas (Cierres y Aperturas)
    // strel('disk', n) 
    Mat kernel4, kernel6, kernel8;
    kernel4 = getStructuringElement(MORPH_ELLIPSE, Size(9, 9));
    kernel6 = getStructuringElement(MORPH_ELLIPSE, Size(13, 13));
    kernel8 = getStructuringElement(MORPH_ELLIPSE, Size(17, 17));

    // mask = imclose(mask, strel('disk',4));
    morphologyEx(mask, mask, MORPH_CLOSE, kernel4);

    // mask = imopen(mask, strel('disk',6));
    morphologyEx(mask, mask, MORPH_OPEN, kernel6);

    // mask = imclose(mask, strel('disk',8));
    morphologyEx(mask, mask, MORPH_CLOSE, kernel8);

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

Mat Segmentacion::FiltrarObjetoLego(const Mat& mask) {
    // 1. CC = bwconncomp(mask) y regionprops
    // En OpenCV usamos findContours para obtener los objetos
    vector<vector<Point>> contours;
    findContours(mask, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

    if (contours.empty()) return Mat::zeros(mask.size(), CV_8UC1);

    // 2. Calcular áreas y encontrar la máxima
    double maxArea = 0.0;
    for (const auto& contour : contours) 
        maxArea = max(maxArea, contourArea(contour));

    // 3. Crear una máscara vacía (BW = false(size(mask)))
    Mat BW = Mat::zeros(mask.size(), CV_8UC1);

    // 4. Filtrar: props.Area >= 0.1 * max(props.Area)
    double thresholdArea = 0.1 * maxArea;

    for (const auto& contour : contours) {
        if (contourArea(contour) >= thresholdArea) {
            // Dibujar el contorno relleno (equivale a asignar PixelIdxList)
            // -1 significa rellenar el interior del contorno
            drawContours(BW, vector<vector<Point>>{contour}, -1, Scalar(255), FILLED);
        }
    }

    return BW;
}

Segmentacion::ObjetosSegmentados Segmentacion::RecorteAjusteImagen(const vector<Mat> &hsv_channels, const  Mat& BW) {

    ObjetosSegmentados resultado;
    Size tam_final(200, 200);
    int extra = 20;
 
	Mat I_hsv_float, I_bgr_float,I_original;
	merge(hsv_channels, I_hsv_float);
	
    cvtColor(I_hsv_float, I_bgr_float, COLOR_HSV2BGR);
    I_bgr_float.convertTo(I_original, CV_8UC3, 255.0);

    Mat bw_8u;
    BW.convertTo(bw_8u, CV_8UC1, 255.0);
    // SE = strel('square', 15);
    Mat SE = getStructuringElement(MORPH_RECT, Size(15, 15));

    // Encontrar componentes (props en MATLAB)
    vector<vector<Point>> contours;
    findContours(bw_8u, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

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
        Mat mask_local = Mat::zeros(bw_8u.size(), CV_8UC1);
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

vector<vector<Point>> Segmentacion::MostrarBordes(const Mat& bw) {
    // 1. Preparar la imagen de salida
    // Si I_original es CV_64FC3, la pasamos a 8 bits para visualizar fácilmente
    //Mat res;
    //if (I_original.type() == CV_32FC3)
    //    I_original.convertTo(res, CV_8UC3, 255.0);
    //else if (I_original.type() == CV_8UC3)
    //    res = I_original.clone();
    //else
    //    CV_Error(Error::StsUnsupportedFormat,
    //        "Formato de imagen no soportado");

    // 2. Encontrar los contornos de la máscara
    // Usamos una copia de mask porque findContours puede modificarla
    vector<vector<Point>> contours;
    findContours(bw.clone(), contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

    // 3. Dibujar los bordes sobre la imagen
    // Scalar(0, 255, 0) dibujará el borde en VERDE
    // El parámetro '2' es el grosor de la línea
   

    return contours;
}

