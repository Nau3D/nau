#include "dlgDbgBuffers.h"
#include "../glInfo.h"
#include <nau.h>
#include <nau/debug/profile.h>

BEGIN_EVENT_TABLE(DlgDbgBuffers, wxDialog)
	EVT_PG_CHANGED(DLG_MI_PGBUFFERS, DlgDbgBuffers::OnBufferSettingsChange)
END_EVENT_TABLE()


wxWindow *DlgDbgBuffers::m_Parent = NULL;
DlgDbgBuffers *DlgDbgBuffers::m_Inst = NULL;



void 
DlgDbgBuffers::SetParent(wxWindow *p) {

	m_Parent = p;
}


DlgDbgBuffers* 
DlgDbgBuffers::Instance () {

	if (m_Inst == NULL)
		m_Inst = new DlgDbgBuffers();

	return m_Inst;
}
 

DlgDbgBuffers::DlgDbgBuffers(): wxDialog(DlgDbgBuffers::m_Parent, -1, wxT("Nau - Buffers Information"),wxDefaultPosition,
						   wxDefaultSize,wxRESIZE_BORDER|wxDEFAULT_DIALOG_STYLE)
{
	this->SetSizeHints( wxDefaultSize, wxDefaultSize);

	wxBoxSizer *bSizer1;
	bSizer1 = new wxBoxSizer( wxVERTICAL);

	wxStaticBoxSizer * sbSizer1;
	sbSizer1 = new wxStaticBoxSizer( new wxStaticBox( this, wxID_ANY, wxEmptyString ), wxVERTICAL );

	wxNotebook *notebook = new wxNotebook(this, -1);
	//wxNotebookSizer *nbSizer = new wxNotebookSizer(notebook);

	/* Buffers */
	wxPanel *pBuffers = new wxPanel(notebook, -1);
	notebook->AddPage(pBuffers, wxT("Buffers"));

	wxSizer *buffers = new wxBoxSizer(wxVERTICAL);


	pBuffers->SetAutoLayout(TRUE);
	pBuffers->SetSizer(buffers);

	pgBuffers = new wxPropertyGridManager(pBuffers, DLG_MI_PGBUFFERS,
		wxDefaultPosition, wxDefaultSize,
		// These and other similar styles are automatically
		// passed to the embedded wxPropertyGrid.
		wxPG_BOLD_MODIFIED | wxPG_SPLITTER_AUTO_CENTER |
		// Plus defaults.
		wxPGMAN_DEFAULT_STYLE
		);

	pgBuffers->AddPage(wxT("Buffers"));

	buffers->Add(pgBuffers, 1, wxEXPAND);

	/* VAOs */
	wxPanel *pVAOs = new wxPanel(notebook, -1);
	notebook->AddPage(pVAOs, wxT("VAOs"));

	wxSizer *vaos = new wxBoxSizer(wxVERTICAL);

	//	setupShaders();

	pVAOs->SetAutoLayout(TRUE);
	pVAOs->SetSizer(vaos);

	pgVAOs = new wxPropertyGridManager(pVAOs, DLG_MI_PGVAOS,
		wxDefaultPosition, wxDefaultSize,
		// These and other similar styles are automatically
		// passed to the embedded wxPropertyGrid.
		wxPG_BOLD_MODIFIED | wxPG_SPLITTER_AUTO_CENTER |
		// Plus defaults.
		wxPGMAN_DEFAULT_STYLE
		);

	pgVAOs->AddPage(wxT("VAO"));

	vaos->Add(pgVAOs, 1, wxEXPAND);



	//pgVAOs->Append(pid);
	//pgVAOs->AppendIn(pid, new wxColourProperty(wxT("TESTProp3"), wxPG_LABEL,
	//	wxColour(255, 255, 255)));
	//pgVAOs->AppendIn(pid, new wxFloatProperty(wxT("TESTProp4"), wxPG_LABEL, 1.0));
	//pgVAOs->Expand(pid);

	//Sizer finishing touches

	sbSizer1->Add(notebook, 1, wxALL | wxEXPAND, 5);
	
	bSizer1->Add(sbSizer1, 1, wxEXPAND, 5);

	this->SetSizer(bSizer1);
	this->Layout();
	this->Centre(wxBOTH);

	clear();
}


void
DlgDbgBuffers::updateDlg() 
{

}


std::string &
DlgDbgBuffers::getName ()
{
	name = "DlgDbgBuffers";
	return(name);
}


void
DlgDbgBuffers::eventReceived(const std::string &sender, const std::string &eventType, nau::event_::IEventData *evt)
{
}


void DlgDbgBuffers::append(std::string s) {

}


void DlgDbgBuffers::clear() {

	isLogClear = true;

	pgBuffers->ClearPage(0);

	pgVAOs->ClearPage(0);
}

void DlgDbgBuffers::loadBufferInfo() {
	
	if (isLogClear){
		std::vector<std::pair<std::pair<int, int>, std::vector<int>>> vaoInfoData;

		getCurrentVAOInfoData(vaoInfoData);

		for (std::pair<std::pair<int, int>, std::vector<int>> vaoInfo : vaoInfoData){
			wxPGProperty *pid;
			wxPGProperty *appended, *buffers;

			pid = pgVAOs->Append(new wxPGProperty(wxT("VAO " + std::to_string(vaoInfo.first.first)), wxPG_LABEL));

			appended = pgVAOs->AppendIn(pid, new wxIntProperty(wxT("Element Array"), wxPG_LABEL,
				vaoInfo.first.second));
			appended->Enable(false);

			buffers = pgVAOs->AppendIn(pid, new wxPGProperty(wxT("Buffers"), wxPG_LABEL));
			//pid->Enable(false);
			for (int bufferName : vaoInfo.second){
				appended = pgVAOs->AppendIn(buffers, new wxIntProperty(wxT("Buffer"), wxPG_LABEL,
					bufferName));
				appended->Enable(false);
			}
			pgVAOs->Expand(pid);
		}
		loadBufferInfoGrid();
	}
}


void DlgDbgBuffers::loadBufferInfoGrid(){
	std::vector<int> bufferList = getCurrentBufferNames();


	for (int buffer : bufferList){
		std::pair<std::vector<int>, std::vector<std::string>> bufferInfo;
		bool defined = getBufferInfo(buffer, bufferInfo);
		wxPGProperty *pid, *values, *valueline, *appended;

		//info order:
		//bufferattribute.push_back("Size: " + std::to_string(info));
		//bufferattribute.push_back("Components: " + std::to_string(info));
		//bufferattribute.push_back("Type: " + spDataF[info]);
		//bufferattribute.push_back("Stride: " + std::to_string(info));
		//bufferattribute.push_back("Normalized: " + std::to_string(info));
		//bufferattribute.push_back("Divisor: " + std::to_string(info));
		//bufferattribute.push_back("Integer: " + std::to_string(info));

		pid = pgBuffers->Append(new wxPGProperty(wxT("Buffer " + std::to_string(buffer)), wxPG_LABEL));
		pid->SetAttribute(wxT("buffer"), wxVariant(buffer));
		if (defined){
			std::string valueHeader;
			appended = pgVAOs->AppendIn(pid, new wxIntProperty(wxT("Size (Real)"), wxPG_LABEL, bufferInfo.first[0]));
			appended->Enable(false);
			appended = pgVAOs->AppendIn(pid, new wxIntProperty(wxT("Length"), wxPG_LABEL, bufferInfo.second.size()));
			appended->Enable(false);
			appended = pgVAOs->AppendIn(pid, new wxIntProperty(wxT("Components"), wxPG_LABEL, bufferInfo.first[1]));
			appended->Enable(false);
			appended = pgVAOs->AppendIn(pid, new wxStringProperty(wxT("Type"), wxPG_LABEL, getDatatypeString(bufferInfo.first[2])));
			appended->Enable(false);
			appended = pgVAOs->AppendIn(pid, new wxIntProperty(wxT("Stride"), wxPG_LABEL, bufferInfo.first[3]));
			appended->Enable(false);
			appended = pgVAOs->AppendIn(pid, new wxIntProperty(wxT("Normalized"), wxPG_LABEL, bufferInfo.first[4]));
			appended->Enable(false);
			appended = pgVAOs->AppendIn(pid, new wxIntProperty(wxT("Divisor"), wxPG_LABEL, bufferInfo.first[5]));
			appended->Enable(false);
			appended = pgVAOs->AppendIn(pid, new wxIntProperty(wxT("Integer"), wxPG_LABEL, bufferInfo.first[6]));
			appended->Enable(false);
			values = pgVAOs->AppendIn(pid, new wxPGProperty(wxT("Values"), wxPG_LABEL));
			values->Enable(false);
			valueline = 0;
			for (int valueNum = 0; valueNum < bufferInfo.second.size(); valueNum++){
				if (valueNum % 4 == 0){
					if (valueNum + 3 < bufferInfo.second.size()){
						valueHeader = "[" + std::to_string(valueNum + 1) + " <-> " + std::to_string(valueNum + 4) + "]";
					}
					else{
						valueHeader = "[" + std::to_string(valueNum + 1) + " <-> " + std::to_string(bufferInfo.second.size()) + "]";
					}
					valueline = pgVAOs->AppendIn(values, new wxStringProperty(valueHeader, wxPG_LABEL, wxT("<composed>")));
					valueline->Enable(false);
				}
				appended = pgVAOs->AppendIn(valueline, new wxStringProperty(wxT("[" + std::to_string(valueNum + 1) + "]"), wxPG_LABEL, bufferInfo.second[valueNum]));
				appended->Enable(false);
			}
			pid->Enable(false);
		}
		else{
			appended = pgVAOs->AppendIn(pid, new wxIntProperty(wxT("Size (Real)"), wxPG_LABEL, bufferInfo.first[0]));
			appended->Enable(false);
			appended = pgVAOs->AppendIn(pid, new wxIntProperty(wxT("Size (Types)"), wxPG_LABEL, 0));
			appended->Enable(false);
			appended = pgVAOs->AppendIn(pid, new wxIntProperty(wxT("Length"), wxPG_LABEL, 0));
			appended = pgVAOs->AppendIn(pid, new wxStringProperty(wxT("Type"), wxPG_LABEL, wxT("<composed>")));
			appended = pgVAOs->AppendIn(pid, new wxPGProperty(wxT("Values"), wxPG_LABEL));
			appended->Enable(false);

		}
	}
	pgBuffers->CollapseAll();
}

void DlgDbgBuffers::OnBufferSettingsChange(wxPropertyGridEvent& e){
	wxPGProperty *eventProperty = e.GetProperty();
	wxString name = e.GetProperty()->GetLabel();
	if (name.StartsWith("Length")){
		wxIntProperty *length = (wxIntProperty*)eventProperty;
		wxPGProperty *typelist;
		unsigned int currentTypesCount;
		long l = length->GetValue().GetLong();

		if (l < 0){
			length->SetValueFromInt(0);
		}

		typelist = length->GetParent()->GetPropertyByName("Type");
		currentTypesCount = typelist->GetChildCount();
		if (l > currentTypesCount){
			const wxChar* dataType[] = { wxT("BYTE"), wxT("UNSIGNED_BYTE"), wxT("INT"), wxT("UNSIGNED_INT"), wxT("SHORT"), wxT("UNSIGNED_SHORT"), wxT("FLOAT"), wxT("DOUBLE"), NULL };
			const long dataTypeInd[] = { DLG_BYTE, DLG_UNSIGNED_BYTE, DLG_INT, DLG_UNSIGNED_INT, DLG_SHORT, DLG_UNSIGNED_SHORT, DLG_FLOAT, DLG_DOUBLE };

			for (int i = currentTypesCount; i < l; i++){
				pgBuffers->AppendIn(typelist, new wxEnumProperty(wxT("Type " + std::to_string(i + 1)), wxPG_LABEL, dataType, dataTypeInd, DLG_BYTE));
			}
			pgBuffers->Collapse(typelist);
		}
		else if (currentTypesCount > l){
			while (currentTypesCount > l){
				wxPGProperty *child = typelist->GetPropertyByName("Type " + std::to_string(currentTypesCount));
				pgBuffers->DeleteProperty(child);
				currentTypesCount--;
			}
			typelist->RefreshChildren();
		}
		else if (l <= 0){
			typelist->DeleteChildren();
		}
		updateBufferData(eventProperty->GetParent(), l);
	} 
	else if (name.StartsWith("Type ")){
		updateBufferData(eventProperty->GetParent()->GetParent());
	}
}

void DlgDbgBuffers::updateBufferData(wxPGProperty *bufferProperty, int currentTypeCount){
	int totalsize = 0, size, prevBuffer;
	long realSize;
	std::vector<int> sizes;
	std::vector<void*> pointers;
	std::vector<DataTypes> types;
	wxPGProperty *sizeProperty = bufferProperty->GetPropertyByName("Size (Types)");
	wxPGProperty *sizeRealProperty = bufferProperty->GetPropertyByName("Size (Real)");
	wxPGProperty *typeProperty = bufferProperty->GetPropertyByName("Type");
	wxPGProperty *valuesProperty = bufferProperty->GetPropertyByName("Values");

	long buffer = bufferProperty->GetAttribute(wxT("buffer")).GetLong();

	realSize = sizeRealProperty->GetValue().GetLong();

	if (currentTypeCount < 0){
		currentTypeCount = typeProperty->GetChildCount();
	}

	for (int t = 1; t <= currentTypeCount; t++){
		DataTypes enumValue = getBufferDataType(typeProperty, t);
		size = getBufferDataTypeSize(enumValue);
		totalsize += size;
		sizes.push_back(size);
		types.push_back(enumValue);
	}

	sizeProperty->SetValueFromInt(totalsize);

	prevBuffer = getOpenBufferMapPointers(buffer, realSize, sizes, pointers);
	updateBufferDataValues(valuesProperty, pointers, types);
	if (realSize < totalsize){
		pgVAOs->AppendIn(valuesProperty, new wxStringProperty(wxT("ERROR!"), wxPG_LABEL, wxT("User size exceeds buffer size!")))->Enable(false);
	}

	getCloseBufferMapPointers(prevBuffer);
}

void DlgDbgBuffers::updateBufferDataValues(wxPGProperty *valuesProperty, std::vector<void*> &pointers, std::vector<DataTypes> types){
	wxPGProperty *valueline, *appended;
	std::string valueHeader;

	valuesProperty->DeleteChildren();

	valueline = 0;
	for (int valueNum = 0; valueNum < pointers.size(); valueNum++){
		if (valueNum % 4 == 0){
			if (valueline){
				pgVAOs->Collapse(valueline);
			}
			if (valueNum + 3 < pointers.size()){
				valueHeader = "[" + std::to_string(valueNum + 1) + " <-> " + std::to_string(valueNum + 4) + "]";
			}
			else{
				valueHeader = "[" + std::to_string(valueNum + 1) + " <-> " + std::to_string(pointers.size()) + "]";
			}
			valueline = pgVAOs->AppendIn(valuesProperty, new wxStringProperty(valueHeader, wxPG_LABEL, wxT("<composed>")));
			valueline->Enable(false);
		}
		appended = pgVAOs->AppendIn(valueline, new wxStringProperty(wxT("[" + std::to_string(valueNum + 1) + "]"), wxPG_LABEL, getStringFromPointer(types[valueNum], pointers[valueNum])));
		appended->Enable(false);
	}
	if (valueline){
		pgVAOs->Collapse(valueline);
	}
	pgBuffers->Collapse(valuesProperty);
}

DlgDbgBuffers::DataTypes DlgDbgBuffers::getBufferDataType(wxPGProperty *typeProperty, int index){
	wxVariant type = typeProperty->GetPropertyByName("Type " + std::to_string(index))->GetValue();
	return static_cast<DataTypes>(type.GetLong());
}

int DlgDbgBuffers::getBufferDataTypeSize(DataTypes type){
	switch (type){
	case DLG_BYTE:
	case DLG_UNSIGNED_BYTE:
		return sizeof(char);
	case DLG_INT:
	case DLG_UNSIGNED_INT:
		return sizeof(int);
	case DLG_SHORT:
	case DLG_UNSIGNED_SHORT:
		return sizeof(short);
	case DLG_FLOAT:
		return sizeof(float);
	case DLG_DOUBLE:
		return sizeof(double);
	}
	return 0;
}



std::string DlgDbgBuffers::getStringFromPointer(DataTypes type, void* ptr){
	int intconverter;
	unsigned int uintconverter;
	//try{
		switch (type){
		case DLG_BYTE:
			intconverter = *((char*)ptr);
			return std::to_string(intconverter);
		case DLG_UNSIGNED_BYTE:
			uintconverter = *((unsigned char*)ptr);
			return std::to_string(uintconverter);
		case DLG_INT:
			intconverter = *((int*)ptr);
			return std::to_string(intconverter);
		case DLG_UNSIGNED_INT:
			uintconverter = *((unsigned int*)ptr);
			return std::to_string(uintconverter);
		case DLG_SHORT:
			return std::to_string(*((short*)ptr));
		case DLG_UNSIGNED_SHORT:
			return std::to_string(*((unsigned short*)ptr));
		case DLG_FLOAT:
			return std::to_string(*((float*)ptr));
		case DLG_DOUBLE:
			return std::to_string(*((double*)ptr));
		}
	//}
	//catch(){
	//	return "ERROR!";
	//}
	return "unknown";
}