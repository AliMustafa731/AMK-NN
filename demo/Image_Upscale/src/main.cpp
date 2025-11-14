
#include <wx/wx.h>
#include "components/MainWindow.h"


class MyApp : public wxApp
{
public:
    bool OnInit()
    {
        MainWindow *frame = new MainWindow("AMK-NN: Image Upscale", wxPoint(50, 50), wxSize(960, 640));
        frame->Show(true);
        frame->Center();
        return true;
    }
};

// Create the application instance
wxIMPLEMENT_APP(MyApp);
