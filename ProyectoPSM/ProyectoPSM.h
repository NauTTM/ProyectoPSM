#ifndef PROYECTOPSM_H
#define PROYECTOPSM_H

#include <QMainWindow>
#include <QTimer>
#include <QImage>
#include <QPixmap>
#include <QDir>
#include <QMessageBox>
#include <opencv2/opencv.hpp>
#include "ui_ProyectoPSM.h" 

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
    void iniciarGrabacion();
    void detenerGrabacion();
    void capturarImagen();
    void actualizarFrame();

private:
    Ui::ProyectoPSM* ui;
    cv::VideoCapture camara;
    QTimer* temporizador;
    cv::Mat frameActual;

    void inicializarCombos();
    QString generarNombreArchivo();
    static QImage matToQImage(const cv::Mat& mat);
};

#endif // PROYECTOPSM_H
