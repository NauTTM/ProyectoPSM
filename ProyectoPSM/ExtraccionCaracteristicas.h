#pragma once
#include <QObject>
#include <QtCore>
#include "opencv2/opencv.hpp"
#include <fstream>

using namespace cv;
using namespace std;

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
	void ExtraerXyGClasificacion();
 
private:
	RGBResult obtenerMedianaMediaRGB(const Mat& ImagenSegmentadaColor);
	MomentosHu obtenerMomentosHu(const Mat& BW);
	ImagenProps obtenerPropiedadesImagen(const Mat& BW);

public slots:
	VectorCaracteristicas ExtraerCaracteristicasImagen(const Mat& ImagenSegmentadaColorTamanoAjustado);

signals:
	void ListaCaracterisiticas(const VectorCaracteristicas& caracteristicas);
	
};