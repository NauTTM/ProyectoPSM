#include "ProyectoPSM.h"
#include "ui_ProyectoPSM.h"

ProyectoPSM::ProyectoPSM(QWidget* parent)
    : QMainWindow(parent), ui(new Ui::ProyectoPSM)
{
    ui->setupUi(this);

    // Inicializamos los combos y el temporizador
    inicializarCombos();
    temporizador = new QTimer(this);

    // Conectamos señales y slots
    connect(ui->btnStart, SIGNAL(clicked()), this, SLOT(iniciarGrabacion()));
    connect(ui->btnStop, SIGNAL(clicked()), this, SLOT(detenerGrabacion()));
    connect(ui->btnCapture, SIGNAL(clicked()), this, SLOT(capturarImagen()));
    connect(temporizador, SIGNAL(timeout()), this, SLOT(actualizarFrame()));

    // Crear carpeta dataset si no existe
    QDir dir;
    if (!dir.exists("dataset")) dir.mkdir("dataset");
}

ProyectoPSM::~ProyectoPSM()
{
    if (camara.isOpened())
        camara.release();
    delete ui;
}

void ProyectoPSM::inicializarCombos()
{
    // Códigos (01..20)
    for (int i = 1; i <= 20; i++)
        ui->comboCodigo->addItem(QString("%1").arg(i, 2, 10, QChar('0')));

    // Azimuth (000,045,...,315)
    QStringList az = { "000","045","090","135","180","225","270","315" };
    ui->comboAzimuth->addItems(az);

    // Elevación (010,020,...,090)
    for (int e = 10; e <= 90; e += 10)
        ui->comboElev->addItem(QString("%1").arg(e, 3, 10, QChar('0')));

    // Secuencia (001..999)
    for (int s = 1; s <= 20; s++)
        ui->comboSeq->addItem(QString("%1").arg(s, 3, 10, QChar('0')));
}

void ProyectoPSM::iniciarGrabacion()
{
    // Abrimos cámara
    if (!camara.open(0)) {
        QMessageBox::warning(this, "Error", "No se pudo abrir la cámara.");
        return;
    }

    temporizador->start(30);  // 30 ms ~ 33 fps
}

void ProyectoPSM::detenerGrabacion()
{
    temporizador->stop();
    if (camara.isOpened())
        camara.release();
}

void ProyectoPSM::actualizarFrame()
{
    camara >> frameActual;
    if (frameActual.empty()) return;

    // Convertimos a QImage para mostrar
    QImage imagen = matToQImage(frameActual);
    ui->labelPreview->setPixmap(QPixmap::fromImage(imagen).scaled(
        ui->labelPreview->size(), Qt::KeepAspectRatio, Qt::SmoothTransformation));
}

void ProyectoPSM::capturarImagen()
{
    if (frameActual.empty()) {
        QMessageBox::warning(this, "Captura", "No hay imagen disponible.");
        return;
    }

    QString nombre = generarNombreArchivo();
    cv::imwrite(nombre.toStdString(), frameActual);
    QMessageBox::information(this, "Imagen guardada", nombre);
}

QString ProyectoPSM::generarNombreArchivo()
{
    QString codigo = ui->comboCodigo->currentText();
    QString az = ui->comboAzimuth->currentText();
    QString el = ui->comboElev->currentText();
    QString seq = ui->comboSeq->currentText();
    return QString("dataset/%1_%2_%3_%4.png").arg(codigo, az, el, seq);
}

QImage ProyectoPSM::matToQImage(const cv::Mat& mat)
{
    if (mat.type() == CV_8UC3) {
        cv::Mat rgb;
        cv::cvtColor(mat, rgb, cv::COLOR_BGR2RGB);
        return QImage(rgb.data, rgb.cols, rgb.rows, rgb.step, QImage::Format_RGB888).copy();
    }
    else if (mat.type() == CV_8UC1) {
        return QImage(mat.data, mat.cols, mat.rows, mat.step, QImage::Format_Grayscale8).copy();
    }
    else {
        return QImage();
    }
}
