#pragma once

#include <wx/wx.h>
#include "Image.h"

class MainWindow : public wxFrame
{
public:
    MainWindow(const wxString& title, const wxPoint& pos, const wxSize& size);

private:
    void CreateMenu();
    void CreateControls();
    void SetupSizers();

    // Event handlers
    void OnExit(wxCommandEvent& event);
    void OnAbout(wxCommandEvent& event);
    void OnOpen(wxCommandEvent& event);
    void OnSave(wxCommandEvent& event);

    wxPanel* m_panel;
    wxButton* m_button_train;
    wxButton* m_button_upscale;
    wxButton* m_button_random_select;
    wxChoice* m_choice_dataset;
    wxChoice* m_choice_optimizer;
    wxSlider* m_slider_learning_rate;
    Image* m_image_desired;
    Image* m_image_output;

    wxDECLARE_EVENT_TABLE(); // Declare event table
};
