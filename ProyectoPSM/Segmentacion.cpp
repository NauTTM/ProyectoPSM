#include "Segmentacion.h"

Segmentacion::Segmentacion() {
}

Segmentacion::~Segmentacion() {
}
 /*void Segmentacion::SegmentarTodasImagenes() {
	vector<vector<double>> X;
	vector<double> G;
	QDir directory("imagenes/");

	QStringList filters;
	filters << "*.jpg" << "*.png" << "*.jpeg" << "*.tif";
	QStringList files = directory.entryList(filters, QDir::Files);

    QDir outDir(directory.absolutePath());
    if (!outDir.exists("imagenesSegmentadasSinResize")) {
        outDir.mkdir("imagenesSegmentadasSinResize");
    }
	for (int i = 0; i < files.size(); ++i) {
		QString fileName = files[i];
		int numero = fileName.section('_', 0, 0).toInt();

		Mat I = imread(directory.absoluteFilePath(fileName).toStdString());
		
		Mat imagenSegmentadaRecortada = SegmentarImagen(I);
	
        QFileInfo info(fileName);
        QString baseName = info.completeBaseName(); 
        QString extension = info.suffix();          

        QString newFileName = baseName + "_bw." + extension;

        QString outputPath = outDir.absoluteFilePath("imagenesSegmentadasSinResize/" + newFileName);

        imwrite(outputPath.toStdString(), imagenSegmentadaRecortada);
	}

}*/

// Funcion SegmentarImagen que se ejecuta en un hilo independiente
 void Segmentacion::SegmentarImagen(const Mat& Imagen) {
        // Variables auxiliares
    //Mat im = imread("capturas/captura_20260107_183433.jpg");
        Mat sat, img1, img8bit;
		vector<vector<Point>> img2;

        // Aplicamos balance de blancos
        Mat blanco = BalanceBlancos(Imagen);

        // Conversion a HSV y aumento de saturacion
        vector<Mat> vecSat = AumentoSaturacion(blanco);

        // Correcion de iluminacion
        vector<Mat> corr = CorreccionIluminacion(vecSat);

        // Segmentacion binaria
        Mat mask = SegmentacionImagen(corr);

        // Filtrado del objeto relevante
        Mat bw = FiltrarObjetoLego(mask);

        // Recorte y normalizacion del objeto
        ImagenesSegmentadas imagenes = RecorteAjusteImagen(corr, bw);
        //img1 = objeto; 
        img2 = MostrarBordes(bw); // Obtencion de contornos
        
        // Emision del resultado
        if(!imagenes.objeto.empty())
           emit SegmentacionCompletada(imagenes.objeto, img2, imagenes.imagenesBinarias, imagenes.imagenesColor);
       
}

// Funcion balanceo de blancos para reducir dominantes de color por iluminacion en la imagen
Mat Segmentacion::BalanceBlancos(const Mat& Imagen) {

    //1. Convertir a float y normalizar a [0, 1]
    Mat I_wb;
    Imagen.convertTo(I_wb, CV_32FC3, 1.0 / 255.0);

    //2. Calculo de la media de cada canal
	Scalar meanVal = mean(I_wb);

    //3. Obtencion de las medias por canal 
    float mediaB = meanVal[0];
    float mediaG = meanVal[1];
    float mediaR = meanVal[2];

    //4. Aplicar las correcciones (Evitar división por cero). Canal de refencia: G
    float kR = (mediaR > 0.f) ? (mediaG / mediaR) : 1.f;
    float kB = (mediaB > 0) ? (mediaG / mediaB) : 1.f;

    //5. Aplicacion de las correccion a los canales a los canales R y B
	vector<Mat> canales;
	split(I_wb, canales);
	canales[2] = canales[2] * kR; // Canal R
	canales[0] = canales[0] * kB; // Canal B
	merge(canales, I_wb);
    
    // Retorno
    return I_wb;
}

// Funcion de aumento de saturacion para diferenciar entre objeto y fondo
vector<Mat> Segmentacion::AumentoSaturacion(const Mat& I_wb) {
    // 1. Conversion al espacio HSV
    Mat hsvI;

    // 2. Separacion de los canales HSV
    cvtColor(I_wb, hsvI, COLOR_BGR2HSV);

    // 3. Separar los canales HSV
    vector<Mat> hsv_channels;
    split(hsvI, hsv_channels);


    // 4. Aumento del canal de saturacion 
    hsv_channels[1] *= 1.55f;
    min(hsv_channels[1], 1.0f, hsv_channels[1]); // Limitar a 1.0

    // Retorno
    return hsv_channels;
}

// Funcion para la correccion de iluminacion y corregir sombras y focos de luz
vector<Mat> Segmentacion::CorreccionIluminacion(const vector<Mat> &hsv_channels) {
    
    // Parametros
    Mat V, V_bg, V_small, V_bg_small;

    //1. Seleccion del canal V
	V = hsv_channels[2];

    //2. Reducir solo para calcular el fondo (1/4 del tamaño)
    resize(V, V_small, Size(), 0.25, 0.25, INTER_LINEAR);

    //3. Suavizado fuerte segune estimacion del fondo
    GaussianBlur(V_small, V_bg_small, Size(0, 0), 50.0); // Filtro gaussiano de sigma alto para eliminar estructuras pequenas

    //4. Regresar al tamaño original
    resize(V_bg_small, V_bg, V.size(), 0, 0, INTER_LINEAR);
;
    //5. Primero aplicamos el máximo para evitar dividir por valores muy pequeños o cero
    max(V_bg, 0.3, V_bg);

    //6. Normalizacion del brillo dejando un objeto con brillo homogeneo
    Mat V_corregido;
    divide(V, V_bg, V_corregido);

    //7. Saturacion de valores
	min(V_corregido, 1.0, V_corregido); // Limitar a 1.0

    //8. Reconstruccion del espacio HSV
	vector<Mat> hsv_channels_corregidos = hsv_channels;
	hsv_channels_corregidos[2] = V_corregido;

    //Retorno
    return hsv_channels_corregidos;
}

// Funcion Segmentacion Imagen
Mat Segmentacion::SegmentacionImagen(const vector<Mat> &hsv_channels) {
    // Parametros
    Mat hsv, s_norm, mask;

    // Seleccion del canal de saturacion
    hsv_channels[1].convertTo(s_norm, CV_8UC1, 255.0); 
    
    // Convertimos a 8 bits temporalmente para usar OTSU (equivalente a graythresh)
    Mat s_8u;
    // Umbralizacion automatica OTSU
    threshold(s_norm, mask, 0, 255, THRESH_BINARY | THRESH_OTSU);

    // Operaciones Morfológicas (Cierres y Aperturas) con elempentos estructurantes
    Mat kernel4, kernel6, kernel8;
    kernel4 = getStructuringElement(MORPH_ELLIPSE, Size(9, 9));
    kernel6 = getStructuringElement(MORPH_ELLIPSE, Size(13, 13));
    kernel8 = getStructuringElement(MORPH_ELLIPSE, Size(17, 17));

    // Cierre inicial para relleno de huecos pequenos
    morphologyEx(mask, mask, MORPH_CLOSE, kernel4);
 
    // Apetura para la eliminacion de ruido
    morphologyEx(mask, mask, MORPH_OPEN, kernel6);

    // Cierre final para el suavizado del contorno
    morphologyEx(mask, mask, MORPH_CLOSE, kernel8);

    // Relleno de huecos internos
    // OpenCV no tiene imfill directo, se usa floodFill para llenar huecos simulando imfill de MATLAB
    Mat mask_filled;
    copyMakeBorder(mask, mask_filled, 1, 1, 1, 1,
        BORDER_CONSTANT, Scalar(0));

    floodFill(mask_filled, Point(0, 0), Scalar(255));
    bitwise_not(mask_filled, mask_filled);
    mask_filled = mask_filled(Rect(1, 1, mask.cols, mask.rows));
    mask = (mask | mask_filled);

    // Eliminar objetos que tocan el borde
    Mat temp;
    mask.copyTo(temp);

    // Buscar contornos
    std::vector<std::vector<Point>> contours;
    findContours(temp, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

    for (size_t i = 0; i < contours.size(); ++i) {
        // Si el contorno toca el borde de la imagen, eliminarlo
        Rect bbox = boundingRect(contours[i]);
        if (bbox.x == 0 || bbox.y == 0 ||
            bbox.x + bbox.width == mask.cols ||
            bbox.y + bbox.height == mask.rows) {
            drawContours(mask, contours, (int)i, Scalar(0), FILLED);
        }
    }
    return mask;
}

// Funcion Filtrar Objeto para conservar el objeto de interes (LEGO)
Mat Segmentacion::FiltrarObjetoLego(const Mat& mask) {

    // 1. Deteccion de objetos conectados
    vector<vector<Point>> contours;
    findContours(mask, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

    // Error mascara vacia
    if (contours.empty()) return Mat::zeros(mask.size(), CV_8UC1);

    // 2. Calcular areas y encontrar la maxima
    double maxArea = 0.0;
    for (const auto& contour : contours) 
        maxArea = max(maxArea, contourArea(contour));

    // 3. Crear una mascara vacía 
    Mat BW = Mat::zeros(mask.size(), CV_8UC1);

    // 4. Filtrar: quedarnos con objetas de mas de 10% de area maxima
    double thresholdArea = 0.1 * maxArea;

    for (const auto& contour : contours) {
        if (contourArea(contour) >= thresholdArea) {
            // Dibujar el contorno relleno (equivale a asignar PixelIdxList)
            // -1 significa rellenar el interior del contorno
            drawContours(BW, vector<vector<Point>>{contour}, -1, Scalar(255), FILLED);
        }
    }

    // Retorno
    return BW;
}

// Funcion recorte y ajuste de imagen
Segmentacion::ImagenesSegmentadas  Segmentacion::RecorteAjusteImagen(const vector<Mat>& hsv_channels, const  Mat& BW) {

    // Parametros de normalizacion
    vector<Mat> resultado, resultadoBinario, resultadoColorSinResize;
    Size tam_final(200, 200); // Tamano final comun para todos los objetos
    int extra = 20;
    
    // Reconstruccion de la imagen en color
	Mat I_hsv_float, I_bgr_float,I_original;
	merge(hsv_channels, I_hsv_float);
    cvtColor(I_hsv_float, I_bgr_float, COLOR_HSV2BGR);
    I_bgr_float.convertTo(I_original, CV_8UC3, 255.0);

    // Preparacion de la mascara binaria. Asegura el formato
    Mat bw_8u;
    BW.convertTo(bw_8u, CV_8UC1, 255.0);

    // Elemento estructurante para limpieza local
    Mat SE = getStructuringElement(MORPH_RECT, Size(15, 15));

    // Deteccion del objeto principal
    // Localizacion de todos los objetos segmentados
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

        resultado.push_back(I_norm);
		resultadoColorSinResize.push_back(I_crop_masked);
		resultadoBinario.push_back(mask_crop);
    }

    return {resultado, resultadoColorSinResize, resultadoBinario};
}


// Funcion mostrar bordes
vector<vector<Point>> Segmentacion::MostrarBordes(const Mat& bw) {
    
    // Obtencion de contornos externos del objeto
    vector<vector<Point>> contours;
    findContours(bw.clone(), contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

    // Retorno
    return contours;
}

