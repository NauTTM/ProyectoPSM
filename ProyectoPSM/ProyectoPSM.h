#ifndef PROYECTOPSM_H
#define PROYECTOPSM_H

#include <QMainWindow>
#include <QTimer>
#include <QImage>
#include <QMessageBox>
#include "opencv2/opencv.hpp"
#include "ui_ProyectoPSM.h" 
#include <VideoAcquisition.h>
#include <Segmentacion.h>
#include <ExtraccionCaracteristicas.h>
#include <Clasificador.h>
#include <ClasificacionImagen.h>

using namespace cv;
using namespace Qt;

QT_BEGIN_NAMESPACE
namespace Ui { class ProyectoPSM; }
QT_END_NAMESPACE

class ProyectoPSM : public QMainWindow
{
    Q_OBJECT

public:
    ProyectoPSM(QWidget* parent = nullptr);
    ~ProyectoPSM();

private slots:
    void iniciarDetenerGrabacion();
    void actualizarFrame();
    void capturarImagen();

signals:
    void enviarFrame(const Mat& frame);

private:
    Ui::ProyectoPSM* ui;
    CVideoAcquisition camara;
    QTimer* temporizador;
    QString generarNombreArchivo();
    Mat frameActual;
    bool Recording;
    void MostrarImagenSegmentada(const Mat& img1, const vector<vector<Point>>& Bordes);
    static QImage matToQImage(const Mat& mat);
	Segmentacion *segmentacion;
    QThread *threadSegmentacion;
	int ContadorFrames;
	vector<vector<Point>> BordesActuales;
	
    ExtraccionCaracteristicas *extraccionCaracteristicas;
	QThread *extraccionThread;

	Clasificador* clasificador;
	ClasificacionImagen* clasificacionImagen;
};

#endif // PROYECTOPSM_H