#ifndef PROYECTOPSM_H
#define PROYECTOPSM_H

#include <QMainWindow>
#include <QTimer>
#include <QImage>
#include <QMessageBox>
#include "opencv2/opencv.hpp"
#include "ui_ProyectoPSM.h" 
#include <VideoAcquisition.h>
#include <ExtraccionCaracteristicas.h>
#include <Segmentacion.h>

using namespace cv;

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
    //void iniciarDetenerGrabacion();
   /* void actualizarFrame();
    void capturarImagen();
	void extraerCaracteristicas(std::vector<double> props);*/

private:
    Ui::ProyectoPSM* ui;
    /*CVideoAcquisition camara;
    QTimer* temporizador;*/
  /*  QString generarNombreArchivo();*/
    /*Mat frameActual;
    bool Recording;*/

   /* void inicializarCombos();*/
    static QImage matToQImage(const Mat& mat);
    
	ExtraccionCaracteristicas extraccion;
	Segmentacion segmentacion;
    
};

#endif // PROYECTOPSM_H
