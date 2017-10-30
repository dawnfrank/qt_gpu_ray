/********************************************************************************
** Form generated from reading UI file 'mainwindow.ui'
**
** Created by: Qt User Interface Compiler version 5.9.2
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef UI_MAINWINDOW_H
#define UI_MAINWINDOW_H

#include <QtCore/QVariant>
#include <QtWidgets/QAction>
#include <QtWidgets/QApplication>
#include <QtWidgets/QButtonGroup>
#include <QtWidgets/QComboBox>
#include <QtWidgets/QHBoxLayout>
#include <QtWidgets/QHeaderView>
#include <QtWidgets/QLabel>
#include <QtWidgets/QPushButton>
#include <QtWidgets/QSpacerItem>
#include <QtWidgets/QVBoxLayout>
#include <QtWidgets/QWidget>

QT_BEGIN_NAMESPACE

class Ui_widget_2
{
public:
    QVBoxLayout *verticalLayout;
    QHBoxLayout *horizontalLayout;
    QLabel *label;
    QComboBox *sceneComboBox;
    QLabel *label_2;
    QComboBox *cameraComBox;
    QLabel *label_3;
    QComboBox *sampleComboBox;
    QSpacerItem *horizontalSpacer;
    QPushButton *startBtn;
    QWidget *widget;

    void setupUi(QWidget *widget_2)
    {
        if (widget_2->objectName().isEmpty())
            widget_2->setObjectName(QStringLiteral("widget_2"));
        widget_2->resize(818, 646);
        verticalLayout = new QVBoxLayout(widget_2);
        verticalLayout->setObjectName(QStringLiteral("verticalLayout"));
        horizontalLayout = new QHBoxLayout();
        horizontalLayout->setObjectName(QStringLiteral("horizontalLayout"));
        label = new QLabel(widget_2);
        label->setObjectName(QStringLiteral("label"));

        horizontalLayout->addWidget(label);

        sceneComboBox = new QComboBox(widget_2);
        sceneComboBox->setObjectName(QStringLiteral("sceneComboBox"));

        horizontalLayout->addWidget(sceneComboBox);

        label_2 = new QLabel(widget_2);
        label_2->setObjectName(QStringLiteral("label_2"));

        horizontalLayout->addWidget(label_2);

        cameraComBox = new QComboBox(widget_2);
        cameraComBox->setObjectName(QStringLiteral("cameraComBox"));

        horizontalLayout->addWidget(cameraComBox);

        label_3 = new QLabel(widget_2);
        label_3->setObjectName(QStringLiteral("label_3"));

        horizontalLayout->addWidget(label_3);

        sampleComboBox = new QComboBox(widget_2);
        sampleComboBox->setObjectName(QStringLiteral("sampleComboBox"));

        horizontalLayout->addWidget(sampleComboBox);

        horizontalSpacer = new QSpacerItem(40, 20, QSizePolicy::Expanding, QSizePolicy::Minimum);

        horizontalLayout->addItem(horizontalSpacer);

        startBtn = new QPushButton(widget_2);
        startBtn->setObjectName(QStringLiteral("startBtn"));

        horizontalLayout->addWidget(startBtn);


        verticalLayout->addLayout(horizontalLayout);

        widget = new QWidget(widget_2);
        widget->setObjectName(QStringLiteral("widget"));
        QSizePolicy sizePolicy(QSizePolicy::Preferred, QSizePolicy::Expanding);
        sizePolicy.setHorizontalStretch(0);
        sizePolicy.setVerticalStretch(0);
        sizePolicy.setHeightForWidth(widget->sizePolicy().hasHeightForWidth());
        widget->setSizePolicy(sizePolicy);

        verticalLayout->addWidget(widget);


        retranslateUi(widget_2);

        QMetaObject::connectSlotsByName(widget_2);
    } // setupUi

    void retranslateUi(QWidget *widget_2)
    {
        widget_2->setWindowTitle(QApplication::translate("widget_2", "Form", Q_NULLPTR));
        label->setText(QApplication::translate("widget_2", "\345\234\272\346\231\257", Q_NULLPTR));
        label_2->setText(QApplication::translate("widget_2", "\351\225\234\345\244\264\347\247\215\347\261\273", Q_NULLPTR));
        label_3->setText(QApplication::translate("widget_2", "\351\207\207\346\240\267\346\226\271\345\274\217", Q_NULLPTR));
        startBtn->setText(QApplication::translate("widget_2", "\345\274\200\345\247\213", Q_NULLPTR));
    } // retranslateUi

};

namespace Ui {
    class widget_2: public Ui_widget_2 {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_MAINWINDOW_H
