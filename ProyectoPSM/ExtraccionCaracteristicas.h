#pragma once
#include <QObject>
#include <QtCore>
#include "opencv2/opencv.hpp"

class ExtraccionCaracteristicas : public QThread {
	Q_OBJECT

public:
	

	struct RGBResult {
		double R_mediana;
		double G_mediana;
		double B_mediana;
		double R_media;
		double G_media;
		double B_media;
	};
	struct MomentosHu {
		double hu1;
		double hu2;
		double hu3;
	};

	struct ImagenProps {
		double area;
		double circularidad;
		double perimetro;
	};

	struct VectorCaracteristicas {
		RGBResult rgb;
		MomentosHu hu;
		ImagenProps props;
	};

	ExtraccionCaracteristicas();
	~ExtraccionCaracteristicas(void);
 
public slots:
	RGBResult obtenerMedianaMediaRGB(const cv::Mat& ImagenSegmentadaColor);
	MomentosHu obtenerMomentosHu(const cv::Mat& ImagenSegmentada);
	ImagenProps obtenerPropiedadesImagen(const cv::Mat& ImagenSegmentadaColorTamanoAjustado);
	void listaCaracteristicas();


	
};