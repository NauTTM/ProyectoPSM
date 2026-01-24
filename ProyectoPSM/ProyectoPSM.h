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
#include <Orientacion.h>

using namespace cv;
using namespace Qt;

QT_BEGIN_NAMESPACE
namespace Ui { class ProyectoPSM; }
QT_END_NAMESPACE

// Definicion de la clase
class ProyectoPSM : public QMainWindow
{
    Q_OBJECT

public:
    ProyectoPSM(QWidget* parent = nullptr);
    ~ProyectoPSM();

// Definicion de metodos 
private slots:
    void iniciarDetenerGrabacion();
    void actualizarFrame(const Mat& nuevaImagen);
    void capturarImagen();
    void MostrarResultadoEntrenamiento(double precision);;

// Senal
signals:
    void enviarFrame(const Mat& frame);

// Atributos principales
private:
    Ui::ProyectoPSM* ui;
    CVideoAcquisition* camara;
    QThread* camaraThread;
	void IniciarDetenerGrabacionPestanaClasificacion();
	void ActualizarFramePestanaClasificacion();
    QString generarNombreArchivoPestanaClasificacion();
    Mat frameActual;
    Mat frameActualCaptura;
    bool Recording;
    void MostrarImagenSegmentada(const vector<Mat>& img1, const vector<vector<Point>>& Bordes);
    void MostrarClase(const vector<int> tipoClase, vector<int> orientacion);
    static QImage matToQImage(const Mat& mat);
	Segmentacion *segmentacion;
    QThread *threadSegmentacion;

	int ContadorFrames; // Variable auxilar para control de frames
	vector<vector<Point>> BordesActuales;
	
    ExtraccionCaracteristicas *extraccionCaracteristicas;
	QThread *extraccionThread;

	ClasificacionImagen* clasificacionImagen;
    QThread *clasificadorThread;

    vector<int> TiposClase;
    bool MostrarTexto;

    Orientacion* orientacion;
	QThread* orientacionThread;
    vector<int> OrientacionActual;

	void PrepararPestanaClasificacion();

	void PrepararPestanaCapturaImagen();
    void InicializarCombos();
	void IniciarDetenerGrabacionPestanaCaptura();
	void ActualizarFramePestanaCaptura();
    QString generarNombreArchivoCaptura();
    bool RecordingCaptura;

    Point Centroide, PuntoFinal;

	void PrepararPestanaEntrenamiento();
	
    Clasificador* clasificadorModelo;
    QThread* clasificadorModeloThread;

    bool segmentadorOcupado = false;
};

#endif // PROYECTOPSM_H