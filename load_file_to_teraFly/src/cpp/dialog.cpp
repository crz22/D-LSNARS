#include "dialog.h"

void main()
{
    cout<<"using load_swc"<<endl;
}

/*
void LoadSwcTeraFlyDialog::extract() {
    MarkerList markers;
    for (const auto &m: _callback.getLandmarkTeraFly()) {
        Marker marker;
        marker.x = m.x;
        marker.y = m.y;
        marker.z = m.z;
        markers.push_back(marker);
    }
    if (markers.empty()) return;
    bool ok = false;
    QString savePath = QInputDialog::getText(this, "Set save path", "endswith .marker", QLineEdit::Normal, "", &ok);
    if (!ok) return;
    if (savePath.isEmpty()) {
        savePath = input_swc_path1;
    }
    if (!savePath.endsWith(".marker")) {
        savePath.append(".marker");
    }
    saveMarkerFile(savePath, markers);
}

*/