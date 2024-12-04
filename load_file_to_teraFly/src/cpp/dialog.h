#ifndef __DIALOG_H__
#define __DIALOG_H__

#include <QtGui>
#include "v3d_interface.h"
#include <QWidget>
#include <iostream>
#include <QApplication>
#include <QDateTime>
//#include "surf_objs.h"
#include "basic_surf_objs.h"
//#include "ui_tracing_dialog.h"
//#include "ui_python_dialog.h"
//#include "ui_editing_dialog.h"
#include <qtpropertybrowser.h>
#include <qtpropertymanager.h>
#include <qtvariantproperty.h>
#include <qttreepropertybrowser.h>
#define PI 3.1415926

using namespace std;

class test: public QDialog{
    Q_OBJECT
public:
    explicit test(QWidget* parent = nullptr);
    ~test();
};

/* 
class PythonDialog: public QDialog{
Q_OBJECT
public:
    explicit PythonDialog(QWidget* parent = nullptr);
    ~PythonDialog();
    void readFrom(QComboBox* comboBox);
private:
    //Ui::PythonDialog* ui;
    QComboBox* _comboBox;

private slots:
    void save();
    void add();
    void del();
    void up();
    void down();
};

class EditDialog: public QDialog {
    Q_OBJECT
public:
    explicit EditDialog(V3DPluginCallback2& callback, QWidget* parent = nullptr);
    ~EditDialog();
    QPushButton* getDoneButton() {return ui->doneButton;}
    QPushButton* getSkipButton() {return ui->skipButton;}
//    void setCurWindow(v3dhandle window) {pWindow = window;}
//    void setNeuronTree(NeuronTree* p) {neuronTree = p;}
//    void setSurfObjPath(const QString& path) {surfObjPath = path;}
private slots:
    void connectRunning();
    void connectBlocked();
    void connectDone();
    void traceRunning();
    void traceDone();
//    void saveSurfObj();
//    void open3DWindow();
//    void close3DWindow();
private:
//    V3DPluginCallback2& _callback;
    Ui::EditDialog* ui;
//    v3dhandle pWindow;
    NeuronTree neuronTree;
    QString surfObjPath;
};

class NeuronTracingDialog: public QDialog {
    Q_OBJECT
public:
    explicit NeuronTracingDialog(V3DPluginCallback2& callback, QWidget* parent = nullptr);
    ~NeuronTracingDialog();
    bool saveConfiguration(const QString& savePath);
    QString getMethodName() const {return ui->methodBox->currentText();}
    QString getImageFilePath() const {return ui->imagePath->text();}
    QString getMarkerFilePath() const {return ui->markerPath->text();}
    QString getOutputFolderPath() const {return ui->outputPath->text();}
    QString getPythonEnvironment() const {return ui->pythonBox->currentText();}
    bool getTeraflyFlag() const {return ui->teraflyChecker->isChecked();}
    double getConnectDistance() const {return ui->connectionSpinBox->value();}
    int getMethodIndex() const {return ui->methodBox->currentIndex();}
    void setMethodIndex(int i) {ui->methodBox->setCurrentIndex(i);}
    void setImageFilePath(const char* imagePath) {ui->imagePath->setText(imagePath);}
    void setMarkerFilePath(const char* markerPath) {ui->markerPath->setText(markerPath);}
    void setOutputFolderPath(const char* outputPath) {ui->outputPath->setText(outputPath);}
    void setPythonEnvironment(int i) {ui->pythonBox->setCurrentIndex(i);}
    void setTerafly(bool enable) {ui->teraflyChecker->setChecked(enable);}

private:
    Ui::TracingDialog* ui;
    PythonDialog* pythonDialog;
    QtVariantEditorFactory* varFactory;
    QtVariantPropertyManager* propertyManagers;
    QtTreePropertyBrowser* propertyBrowsers;
    QList<QtVariantProperty*> parameterList;
    int currentMethodIndex;

private slots:
    void readParameter(int index);
    void saveParameter();
    void selectImageDir();
    void selectMarker();
    void selectOutput();
    void editPython();
    void readCache();
    void saveCache();
};
 */


//class NeuronReconstructorDialog : public QDialog
//{
//	Q_OBJECT
//
//private:
//    QSpinBox * maSpinbox;
//    QSpinBox * mpSpinbox;
//    QSpinBox * nSpinbox;
//    QSpinBox * nodeStepSpinbox;
//    QSpinBox * expendRatesSpinbox;
//    QSpinBox * shrinkRateSpinbox;
//    QLineEdit * angleT;
//    QSpinBox * maxIterSpinbox;
//    QSpinBox * stepSizeSpinbox;
//    QSpinBox * maskSizeSpinbox;
//    QLineEdit * tCon;
//
//    QSpinBox * rSpinbox;
//    QSpinBox * mcSpinbox;
//    QDoubleSpinBox * connectDist;
//    QLineEdit * pythonEnvironment;
//    QLineEdit * imageFilePath;
//    QLineEdit * markerFilePath;
//    QLineEdit * outputFolderPath;
//
//    QCheckBox * enflagChecker;
//    QCheckBox * jdFlagChecker;
//    QCheckBox * teraflyChecker;
//    QCheckBox * resumeChecker;
//    QCheckBox * segmentChecker;
//    QCheckBox * criticalPointsDetectChecker;
//    QComboBox * methodBox;
//
//    QHBoxLayout * hbox1;
//    QVBoxLayout * vbox;
//    QHBoxLayout * hbox2;
//    QHBoxLayout * hbox3;
//    QHBoxLayout * hbox4;
//    QHBoxLayout * hbox5;
//    QHBoxLayout * hbox6;
//    QPushButton * ok;
//    QPushButton * cancel;
//    QPushButton * selectPythonEnvironmentButton;
//    QPushButton * selectImageFileButton;
//    QPushButton * selectImageDirButton;
//    QPushButton * selectMarkerFileButton;
//    QPushButton * selectOutputFolderButton;
//
//    QGridLayout* layout;
//    V3DPluginCallback2& callback;
//    QList<QLabel*> labelList;
//public:
//    explicit NeuronReconstructorDialog(V3DPluginCallback2 &cb);
//    ~NeuronReconstructorDialog();
//    bool saveConfiguration(QString& configurationPath) const;
//    bool getTeraflyFlag() const {return teraflyChecker->isChecked();}
//    int getMethodIndex() const {return methodBox->currentIndex();}
//    QString getMethodName() const {return methodBox->currentText();}
//    QString getPythonEnvironment() const {return pythonEnvironment->text();}
//    QString getImageFilePath() const {return imageFilePath->text();}
//    QString getMarkerFilePath() const {return markerFilePath->text();}
//    QString getOutputFolderPath() const {return outputFolderPath->text();}
//    double getConnectDistance() const {return connectDist->value();}
//    void setPythonEnvironment(const char* envName) {pythonEnvironment->setText(envName);}
//    void setImageFilePath(const char* imagePath) {imageFilePath->setText(imagePath);}
//    void setMarkerFilePath(const char* markerPath) {markerFilePath->setText(markerPath);}
//    void setOutputFolderPath(const char* outputPath) {outputFolderPath->setText(outputPath);}
//    void setTeraFly(bool enable) {teraflyChecker->setChecked(enable);}
//
//public slots:
//    void selectImageFile();
//    void selectImageDir();
//    void selectMarkerFile();
//    void selectOutputFolder();
//    void selectPythonEnvironment();
//};
//
//class NeuronSegmentorDialog : public QDialog
//{
//	Q_OBJECT
//
//private:
//    QGridLayout *layout;
//    QHBoxLayout *hbox1;
//    QHBoxLayout *hbox2;
//    QHBoxLayout *hbox3;
//    QHBoxLayout *hbox4;
//    QVBoxLayout *vbox1;
//
//    QLineEdit *inimg_path_text;
//	QLineEdit *output_folder_text;
//	QLineEdit *python_environment_text;
//
//	QPushButton *select_inimg_button;
//    QPushButton *select_image_dir_button;
//	QPushButton *select_output_folder_button;
//	QPushButton *select_python_environment_button;
//    QPushButton *ok;
//    QPushButton *cancel;
//
//public:
//	NeuronSegmentorDialog(V3DPluginCallback2& callback);
//	~NeuronSegmentorDialog();
//    QString getImagePath() {return inimg_path_text->text();}
//    QString getOutputFolder() {return output_folder_text->text();}
//    QString getPythonEnvironment() {return python_environment_text->text();}
//
//public slots:
//    void select_inimg();
//    void select_image_dir();
//    void select_output_folder();
//    void select_python_environment();
//};
//
//class CriticalPointsDetectiongDialog : public QDialog
//{
//    Q_OBJECT
//
//public:
//    QGridLayout *layout;
//    QHBoxLayout *hbox1;
//    QHBoxLayout *hbox2;
//    QHBoxLayout *hbox3;
//    QHBoxLayout *hbox4;
//    QVBoxLayout *vbox1;
//
//    QLineEdit *inimg_path_text;
//	QLineEdit *output_folder_text;
//	QLineEdit *python_environment_text;
//
//	QPushButton *select_inimg_button;
//	QPushButton *select_output_folder_button;
//	QPushButton *select_python_environment_button;
//    QPushButton *ok;
//    QPushButton *cancel;
//
//	QString inimg_path,output_folder,python_environment;
//    v3dhandle img_window;
//
//public:
//	CriticalPointsDetectiongDialog(V3DPluginCallback2 &callback);
//	~CriticalPointsDetectiongDialog();
//
//public slots:
//    void update();
//    void select_inimg();
//    void select_output_folder();
//    void select_python_environment();
//};
/*
class LoadSwcTeraFlyDialog: public QDialog
{
    Q_OBJECT

public:
	QLabel *input_swc_path_label1;
    QLineEdit *input_swc_path_text1;
	QLabel *input_swc_path_label2;
	QLineEdit *input_swc_path_text2;
    QLabel *offsetX;
    QLabel *offsetY;
    QLabel *offsetZ;
    QSpinBox* offsetXbox;
    QSpinBox* offsetYbox;
    QSpinBox* offsetZbox;
    QPushButton *selete_button1;
	QPushButton *selete_button2;
    QPushButton *loadSwc1;
	QPushButton *loadSwc2;
    QPushButton *extractMarker;
    QGridLayout *layout;
    QHBoxLayout *hbox1;
    QHBoxLayout *hbox2;
    QHBoxLayout *hbox3;
	QHBoxLayout *hbox4;
	QHBoxLayout *hbox5;
    QString input_swc_path1;
	QString input_swc_path2;
private:
    V3DPluginCallback2& _callback;
public:
    explicit LoadSwcTeraFlyDialog(V3DPluginCallback2& callback, QWidget* parent = nullptr);
    ~LoadSwcTeraFlyDialog();

public Q_SLOTS:
    void update();
    void select_swc1();
	void select_swc2();
    void load1();
	void load2();
    void extract();
};

class CheckLoopDialog: public QDialog
{
    Q_OBJECT
private:
    QLabel* label;
    QLineEdit* lineEdit;
    QPushButton* selectFile;
    QPushButton* selectDir;
    QPushButton* ok;
    QPushButton* cancel;
    QHBoxLayout* hbox;
    QGridLayout* layout;
public:
    CheckLoopDialog();
    ~CheckLoopDialog();
    bool checkLoop(QString& filePath);
    QString getFilePath();
public slots:
    // void update();
    void selectSWCFile();
    void selectSWCDir();
};
*/
#endif//DIALOG_H

