#include <dialogs/propertyManager.h>

#include <assert.h>


void
PropertyManager::createGrid(wxPropertyGridManager *pg, nau::AttribSet &attribs) {

	std::map<std::string, std::unique_ptr<Attribute>> & attributes = attribs.getAttributes();

	for (auto& attrib:attributes) {

		addAttribute(pg, attrib.second);
	}
	pg->CollapseAll();
	pg->Refresh();
}


void
PropertyManager::createOrderedGrid(wxPropertyGridManager *pg, nau::AttribSet &attribs, 
					std::vector<std::string> &list) {

	// First add the attributes in the list
	for (auto name : list) {

		std::unique_ptr<Attribute> &attr = attribs.get(name);
		addAttribute(pg, attr);
	}

	// Add editable attributes
	std::map<std::string, std::unique_ptr<Attribute>> & attributes = attribs.getAttributes();

	for (auto & attrib : attributes) {

		if (!inList(attrib.second->getName(), list) && attrib.second->getReadOnlyFlag() == false)
		addAttribute(pg, attrib.second);
	}

	// Add the editable attributes
	for (auto& attrib : attributes) {

		if (!inList(attrib.second->getName(), list) && attrib.second->getReadOnlyFlag() == true)
			addAttribute(pg, attrib.second);
	}

	pg->CollapseAll();
	pg->Refresh();
}


void
PropertyManager::setAllReadOnly(wxPropertyGridManager *pg, nau::AttribSet &attribs) {

	std::map<std::string, std::unique_ptr<Attribute>> & attributes = attribs.getAttributes();

	wxPGProperty *pid;
	for (auto& attrib:attributes) {
		pid = pg->GetProperty(attrib.first);
		pg->DisableProperty(pid);
	}
	pg->Refresh();
}


bool
PropertyManager::inList(std::string attr, std::vector<std::string> &list) {

	for (auto s : list) {

		if (s == attr)
			return true;
	}
	return false;
}


void 
PropertyManager::addAttribute(wxPropertyGridManager *pg, std::unique_ptr<Attribute> &a) {

	switch (a->getType()) {

	case Enums::ENUM: createEnum(pg, a); break;
	case Enums::BOOL: createBool(pg, a); break;
	case Enums::BVEC4: createBVec4(pg, a); break;
	case Enums::INT: createInt(pg, a); break;
	case Enums::IVEC3: createIVec3(pg, a); break;
	case Enums::UINT: createUInt(pg, a); break;
	case Enums::UIVEC2: createUIVec2(pg, a); break;
	case Enums::UIVEC3: createUIVec3(pg, a); break;
	case Enums::FLOAT: createFloat(pg, a); break;
	case Enums::VEC2: createVec2(pg, a); break;
	case Enums::VEC3: createVec3(pg, a); break;
	case Enums::VEC4: createVec4(pg, a); break;
	case Enums::MAT3: createMat3(pg, a); break;
	case Enums::MAT4: createMat4(pg, a); break;
	default: assert(false && "Missing datatype in property manager");

	}
}


void 
PropertyManager::updateGrid(wxPropertyGridManager *pg, nau::AttribSet &attribs, AttributeValues *attribVal) {

	std::map<std::string, std::unique_ptr<Attribute>> & attributes = attribs.getAttributes();

	pg->ClearSelection();

	for (auto &attrib : attributes) {

		std::unique_ptr<Attribute> &a = attrib.second;
		switch (a->getType()) {

		case Enums::ENUM: updateEnum(pg, a->getName(), attribVal->getPrope((AttributeValues::EnumProperty)a->getId())); break;
		case Enums::BOOL: updateBool(pg, a->getName(), attribVal->getPropb((AttributeValues::BoolProperty)a->getId())); break;
		case Enums::BVEC4: updateBVec4(pg, a->getName(), attribVal->getPropb4((AttributeValues::Bool4Property)a->getId())); break;
		case Enums::INT: updateInt(pg, a->getName(), attribVal->getPropi((AttributeValues::IntProperty)a->getId())); break;
		case Enums::IVEC3: updateIVec3(pg, a->getName(), attribVal->getPropi3((AttributeValues::Int3Property)a->getId())); break;
		case Enums::UINT: updateInt(pg, a->getName(), attribVal->getPropui((AttributeValues::UIntProperty)a->getId())); break;
		case Enums::UIVEC2: updateUIVec2(pg, a->getName(), attribVal->getPropui2((AttributeValues::UInt2Property)a->getId())); break;
		case Enums::UIVEC3: updateUIVec3(pg, a->getName(), attribVal->getPropui3((AttributeValues::UInt3Property)a->getId())); break;
		case Enums::FLOAT: updateFloat(pg, a->getName(), attribVal->getPropf((AttributeValues::FloatProperty)a->getId())); break;
		case Enums::VEC2: updateVec2(pg, a->getName(), attribVal->getPropf2((AttributeValues::Float2Property)a->getId())); break;
		case Enums::VEC3: updateVec3(pg, a->getName(), attribVal->getPropf3((AttributeValues::Float3Property)a->getId())); break;
		case Enums::VEC4: 
			if (a->getSemantics() == Attribute::Semantics::COLOR)
				updateVec4Color(pg, a->getName(), attribVal->getPropf4((AttributeValues::Float4Property)a->getId()));
			else
				updateVec4(pg, a->getName(), attribVal->getPropf4((AttributeValues::Float4Property)a->getId()));
			break;
		case Enums::MAT3: updateMat3(pg, a->getName(), attribVal->getPropm3((AttributeValues::Mat3Property)a->getId())); break;
		case Enums::MAT4: updateMat4(pg, a->getName(), attribVal->getPropm4((AttributeValues::Mat4Property)a->getId())); break;
		default: assert(false && "Missing datatype in property manager");
		}
	}
}


void
PropertyManager::updateProp(wxPropertyGridManager *pg, std::string prop, AttribSet &attribs, AttributeValues *attribVal) {

	std::unique_ptr<Attribute> &a = attribs.get(prop);
	Enums::DataType dt;
	wxPGProperty *pgProp;
	int id,i;
	unsigned int ui; uivec2 uiv2; uivec3 uiv3;
	bool b;
	std::string s;
	attribs.getPropTypeAndId(prop, &dt, &id);
	float f;
	vec2 v; vec3 v3; vec4 v4;
	bvec4 b4;
	mat3 m3; mat4 m4;
	Attribute::Semantics sem = a->getSemantics();
	wxColour col;
	wxVariant variant;
	
	switch (dt) {

	case Enums::ENUM:

		pgProp = pg->GetProperty(wxString(prop));
		i = pgProp->GetValue().GetInteger();
		attribVal->setPrope((AttributeValues::EnumProperty)id, i);
		break;

	case Enums::BOOL:

		pgProp = pg->GetProperty(wxString(prop));
		b = pgProp->GetValue().GetBool();
		attribVal->setPropb((AttributeValues::BoolProperty)id, b);
		break;

	case Enums::BVEC4:

		s = prop + "." + "x";
		pgProp = pg->GetProperty(wxString(s));
		b4.x = pgProp->GetValue().GetBool();

		s = prop + "." + "y";
		pgProp = pg->GetProperty(wxString(s));
		b4.y = pgProp->GetValue().GetBool();

		s = prop + "." + "z";
		pgProp = pg->GetProperty(wxString(s));
		b4.z = pgProp->GetValue().GetBool();

		s = prop + "." + "w";
		pgProp = pg->GetProperty(wxString(s));
		b4.w = pgProp->GetValue().GetBool();
		
		attribVal->setPropb4((AttributeValues::Bool4Property)id, b4);
		break;

	case Enums::INT:

		pgProp = pg->GetProperty(wxString(prop));
		i = pgProp->GetValue().GetInteger();
		attribVal->setPropi((AttributeValues::IntProperty)id, i);
		break;

	case Enums::UINT:

		pgProp = pg->GetProperty(wxString(prop));
		ui = pgProp->GetValue().GetInteger();
		attribVal->setPropui((AttributeValues::UIntProperty)id, ui);
		break;

	case Enums::UIVEC2:

		s = prop + "." + "x";
		pgProp = pg->GetProperty(wxString(s));
		uiv2.x = pgProp->GetValue().GetDouble();
		s = prop + "." + "y";
		pgProp = pg->GetProperty(wxString(s));
		uiv2.y = pgProp->GetValue().GetDouble();
		attribVal->setPropui2((AttributeValues::UInt2Property)id, uiv2);
		break;

	case Enums::UIVEC3:

		s = prop + "." + "x";
		pgProp = pg->GetProperty(wxString(s));
		uiv3.x = pgProp->GetValue().GetDouble();
		s = prop + "." + "y";
		pgProp = pg->GetProperty(wxString(s));
		uiv3.y = pgProp->GetValue().GetDouble();
		s = prop + "." + "w";
		pgProp = pg->GetProperty(wxString(s));
		uiv3.z = pgProp->GetValue().GetDouble();
		attribVal->setPropui3((AttributeValues::UInt3Property)id, uiv3);
		break;

	case Enums::FLOAT:

		pgProp = pg->GetProperty(wxString(prop));
		f = pgProp->GetValue().GetDouble();
		attribVal->setPropf((AttributeValues::FloatProperty)id, f);
		break;

	case Enums::VEC2:

		s = prop + "." + "x";
		pgProp = pg->GetProperty(wxString(s));
		v.x = pgProp->GetValue().GetDouble();
		s = prop + "." + "y";
		pgProp = pg->GetProperty(wxString(s));
		v.y = pgProp->GetValue().GetDouble();
		attribVal->setPropf2((AttributeValues::Float2Property)id, v);
		break;

	case Enums::VEC3:

		s = prop + "." + "x";
		pgProp = pg->GetProperty(wxString(s));
		v3.x = pgProp->GetValue().GetDouble();
		s = prop + "." + "y";
		pgProp = pg->GetProperty(wxString(s));
		v3.y = pgProp->GetValue().GetDouble();
		s = prop + "." + "w";
		pgProp = pg->GetProperty(wxString(s));
		v3.z = pgProp->GetValue().GetDouble();
		attribVal->setPropf3((AttributeValues::Float3Property)id, v3);
		break;

	case Enums::VEC4:
		if (sem == Attribute::COLOR) {
			s = prop + ".RGB";
			variant = pg->GetPropertyValue(wxString(s));
			col << variant;
			v4.x = col.Red()/255.0; v4.y = col.Green() / 255.0; v4.z = col.Blue() / 255.0;
			s = prop + ".Alpha" ;
			pgProp = pg->GetProperty(wxString(s));
			v4.w = pgProp->GetValue().GetDouble();
		}
		else {
			s = prop + "." + "x";
			pgProp = pg->GetProperty(wxString(s));
			v4.x = pgProp->GetValue().GetDouble();
			s = prop + "." + "y";
			pgProp = pg->GetProperty(wxString(s));
			v4.y = pgProp->GetValue().GetDouble();
			s = prop + "." + "z";
			pgProp = pg->GetProperty(wxString(s));
			v4.z = pgProp->GetValue().GetDouble();
			s = prop + "." + "w";
			pgProp = pg->GetProperty(wxString(s));
			v4.w = pgProp->GetValue().GetDouble();
		}
		attribVal->setPropf4((AttributeValues::Float4Property)id, v4);
		break;

	case Enums::MAT3:

		s = prop + "." + "Row0.x";
		pgProp = pg->GetProperty(wxString(s));
		m3.set(0,0, pgProp->GetValue().GetDouble());
		s = prop + "." + "Row0.y";
		pgProp = pg->GetProperty(wxString(s));
		m3.set(0, 1, pgProp->GetValue().GetDouble());
		s = prop + "." + "Row0.z";
		pgProp = pg->GetProperty(wxString(s));
		m3.set(0, 2, pgProp->GetValue().GetDouble());

		s = prop + "." + "Row1.x";
		pgProp = pg->GetProperty(wxString(s));
		m3.set(1, 0, pgProp->GetValue().GetDouble());
		s = prop + "." + "Row1.y";
		pgProp = pg->GetProperty(wxString(s));
		m3.set(1, 1, pgProp->GetValue().GetDouble());
		s = prop + "." + "Row1.z";
		pgProp = pg->GetProperty(wxString(s));
		m3.set(1, 2, pgProp->GetValue().GetDouble());

		s = prop + "." + "Row2.x";
		pgProp = pg->GetProperty(wxString(s));
		m3.set(2, 0, pgProp->GetValue().GetDouble());
		s = prop + "." + "Row2.y";
		pgProp = pg->GetProperty(wxString(s));
		m3.set(2, 1, pgProp->GetValue().GetDouble());
		s = prop + "." + "Row2.z";
		pgProp = pg->GetProperty(wxString(s));
		m3.set(2, 2, pgProp->GetValue().GetDouble());

		attribVal->setPropm3((AttributeValues::Mat3Property)id, m3);
		break;

	case Enums::MAT4:

		s = prop + "." + "Row0.x";
		pgProp = pg->GetProperty(wxString(s));
		m4.set(0, 0, pgProp->GetValue().GetDouble());
		s = prop + "." + "Row0.y";
		pgProp = pg->GetProperty(wxString(s));
		m4.set(0, 1, pgProp->GetValue().GetDouble());
		s = prop + "." + "Row0.z";
		pgProp = pg->GetProperty(wxString(s));
		m4.set(0, 2, pgProp->GetValue().GetDouble());
		s = prop + "." + "Row0.w";
		pgProp = pg->GetProperty(wxString(s));
		m4.set(0, 3, pgProp->GetValue().GetDouble());

		s = prop + "." + "Row1.x";
		pgProp = pg->GetProperty(wxString(s));
		m4.set(1, 0, pgProp->GetValue().GetDouble());
		s = prop + "." + "Row1.y";
		pgProp = pg->GetProperty(wxString(s));
		m4.set(1, 1, pgProp->GetValue().GetDouble());
		s = prop + "." + "Row1.z";
		pgProp = pg->GetProperty(wxString(s));
		m4.set(1, 2, pgProp->GetValue().GetDouble());
		s = prop + "." + "Row1.w";
		pgProp = pg->GetProperty(wxString(s));
		m4.set(1, 3, pgProp->GetValue().GetDouble());

		s = prop + "." + "Row2.x";
		pgProp = pg->GetProperty(wxString(s));
		m4.set(2, 0, pgProp->GetValue().GetDouble());
		s = prop + "." + "Row2.y";
		pgProp = pg->GetProperty(wxString(s));
		m4.set(2, 1, pgProp->GetValue().GetDouble());
		s = prop + "." + "Row2.z";
		pgProp = pg->GetProperty(wxString(s));
		m4.set(2, 2, pgProp->GetValue().GetDouble());
		s = prop + "." + "Row2.w";
		pgProp = pg->GetProperty(wxString(s));
		m4.set(2, 3, pgProp->GetValue().GetDouble());

		s = prop + "." + "Row3.x";
		pgProp = pg->GetProperty(wxString(s));
		m4.set(3, 0, pgProp->GetValue().GetDouble());
		s = prop + "." + "Row3.y";
		pgProp = pg->GetProperty(wxString(s));
		m4.set(3, 1, pgProp->GetValue().GetDouble());
		s = prop + "." + "Row3.z";
		pgProp = pg->GetProperty(wxString(s));
		m4.set(3, 2, pgProp->GetValue().GetDouble());
		s = prop + "." + "Row3.w";
		pgProp = pg->GetProperty(wxString(s));
		m4.set(3, 3, pgProp->GetValue().GetDouble());

		attribVal->setPropm4((AttributeValues::Mat4Property)id, m4);
		break;
	default:
		assert(false && "Missing data type on property manager: updateProp");
	}
}

//		ENUM

void
PropertyManager::createEnum(wxPropertyGridManager *pg, std::unique_ptr<Attribute> &a) {

	wxPGProperty *pid;

	std::vector<std::string> strs;
	a->getOptionStringListSupported( &strs);
	std::vector<int> inds;
	a->getOptionListSupported(&inds);
	wxArrayString arr;
	wxArrayInt ind;
	for (unsigned int i = 0; i < inds.size(); ++i) {
		arr.Add(wxString(strs[i].c_str()));
		ind.Add(inds[i]);
	}

	pid = pg->Append(new wxEnumProperty(wxString(a->getName().c_str()), wxPG_LABEL, arr, ind, inds[0]));

	if (a->getReadOnlyFlag())
		pg->DisableProperty(pid);
}


void
PropertyManager::updateEnum(wxPropertyGridManager *pg, std::string label, int a) {

	pg->SetPropertyValue(wxString(label.c_str()), a);
}


//		BOOL

void
PropertyManager::createBool(wxPropertyGridManager *pg, std::unique_ptr<Attribute> &a) {

	wxPGProperty *pid;
	pid = pg->Append(new wxBoolProperty(wxString(a->getName().c_str()), wxPG_LABEL));
	if (a->getReadOnlyFlag())
		pg->DisableProperty(pid);
}


void
PropertyManager::updateBool(wxPropertyGridManager *pg, std::string label, bool a) {

	pg->SetPropertyValue(wxString(label.c_str()), a);
}


//		 BVEC4

void
PropertyManager::createBVec4(wxPropertyGridManager *pg, std::unique_ptr<Attribute> &a) {

	wxPGProperty* topId;

	topId = pg->Append(new wxStringProperty(wxString(a->getName().c_str()), wxPG_LABEL, wxT("<composed>")));

	pg->AppendIn(topId, new wxBoolProperty(wxT("x"), wxPG_LABEL));
	pg->AppendIn(topId, new wxBoolProperty(wxT("y"), wxPG_LABEL));
	pg->AppendIn(topId, new wxBoolProperty(wxT("z"), wxPG_LABEL));
	pg->AppendIn(topId, new wxBoolProperty(wxT("w"), wxPG_LABEL));

	if (a->getReadOnlyFlag())
		pg->DisableProperty(topId);
}


void
PropertyManager::updateBVec4(wxPropertyGridManager *pg, std::string label, bvec4 a) {

	std::string s = label + '.' + "x";
	pg->SetPropertyValue(wxString(s.c_str()), a.x);
	s.clear();
	s = label + '.' + "y";
	pg->SetPropertyValue(wxString(s.c_str()), a.y);
	s.clear();
	s = label + '.' + "z";
	pg->SetPropertyValue(wxString(s.c_str()), a.z);
	s.clear();
	s = label + '.' + "w";
	pg->SetPropertyValue(wxString(s.c_str()), a.w);
}

//		INT

void
PropertyManager::createInt(wxPropertyGridManager *pg, std::unique_ptr<Attribute> &a) {

	wxPGProperty *pid;
	pid = pg->Append(new wxIntProperty(wxString(a->getName().c_str()), wxPG_LABEL));
	if (a->getReadOnlyFlag())
		pg->DisableProperty(pid);	
}


void
PropertyManager::updateInt(wxPropertyGridManager *pg, std::string label, int a) {

	pg->SetPropertyValue(wxString(label.c_str()), a);
}


//		 IVEC3

void
PropertyManager::createIVec3(wxPropertyGridManager *pg, std::unique_ptr<Attribute> &a) {

	wxPGProperty* topId;

	topId = pg->Append(new wxStringProperty(wxString(a->getName().c_str()), wxPG_LABEL, wxT("<composed>")));

	pg->AppendIn(topId, new wxIntProperty(wxT("x"), wxPG_LABEL));
	pg->AppendIn(topId, new wxIntProperty(wxT("y"), wxPG_LABEL));
	pg->AppendIn(topId, new wxIntProperty(wxT("z"), wxPG_LABEL));

	if (a->getReadOnlyFlag())
		pg->DisableProperty(topId);
}


void
PropertyManager::updateIVec3(wxPropertyGridManager *pg, std::string label, ivec3 a) {

	std::string s = label + '.' + "x";
	pg->SetPropertyValue(wxString(s.c_str()), a.x);
	s.clear();
	s = label + '.' + "y";
	pg->SetPropertyValue(wxString(s.c_str()), a.y);
	s.clear();
	s = label + '.' + "z";
	pg->SetPropertyValue(wxString(s.c_str()), a.z);
}

//		UINT

void
PropertyManager::createUInt(wxPropertyGridManager *pg, std::unique_ptr<Attribute> &a) {

	wxPGProperty *pid;
	pid = pg->Append(new wxUIntProperty(wxString(a->getName().c_str()), wxPG_LABEL));
	if (a->getReadOnlyFlag())
		pg->DisableProperty(pid);
}


void
PropertyManager::updateUInt(wxPropertyGridManager *pg, std::string label, unsigned int a) {

	pg->SetPropertyValue(wxString(label.c_str()), (int)a);
}


//		 IVEC2

void
PropertyManager::createUIVec2(wxPropertyGridManager *pg, std::unique_ptr<Attribute> &a) {

	wxPGProperty* topId;

	topId = pg->Append(new wxStringProperty(wxString(a->getName().c_str()), wxPG_LABEL, wxT("<composed>")));

	pg->AppendIn(topId, new wxUIntProperty(wxT("x"), wxPG_LABEL));
	pg->AppendIn(topId, new wxUIntProperty(wxT("y"), wxPG_LABEL));

	if (a->getReadOnlyFlag())
		pg->DisableProperty(topId);
}


void
PropertyManager::updateUIVec2(wxPropertyGridManager *pg, std::string label, uivec2 a) {

	std::string s = label + '.' + "x";
	pg->SetPropertyValue(wxString(s.c_str()), (int)a.x);
	s.clear();
	s = label + '.' + "y";
	pg->SetPropertyValue(wxString(s.c_str()), (int)a.y);
	s.clear();
}


//		 IVEC3

void
PropertyManager::createUIVec3(wxPropertyGridManager *pg, std::unique_ptr<Attribute> &a) {

	wxPGProperty* topId;

	topId = pg->Append(new wxStringProperty(wxString(a->getName().c_str()), wxPG_LABEL, wxT("<composed>")));

	pg->AppendIn(topId, new wxUIntProperty(wxT("x"), wxPG_LABEL));
	pg->AppendIn(topId, new wxUIntProperty(wxT("y"), wxPG_LABEL));
	pg->AppendIn(topId, new wxUIntProperty(wxT("z"), wxPG_LABEL));

	if (a->getReadOnlyFlag())
		pg->DisableProperty(topId);
}


void
PropertyManager::updateUIVec3(wxPropertyGridManager *pg, std::string label, uivec3 a) {

	std::string s = label + '.' + "x";
	pg->SetPropertyValue(wxString(s.c_str()), (int)a.x);
	s.clear();
	s = label + '.' + "y";
	pg->SetPropertyValue(wxString(s.c_str()), (int)a.y);
	s.clear();
	s = label + '.' + "z";
	pg->SetPropertyValue(wxString(s.c_str()), (int)a.z);
}


//		FLOAT

void
PropertyManager::createFloat(wxPropertyGridManager *pg, std::unique_ptr<Attribute> &a) {

	wxPGProperty *pid;
	pid = pg->Append(new wxFloatProperty(wxString(a->getName().c_str()), wxPG_LABEL));
	if (a->getReadOnlyFlag())
		pg->DisableProperty(pid);
}

void
PropertyManager::updateFloat(wxPropertyGridManager *pg, std::string label, float a) {


	pg->SetPropertyValue(wxString(label.c_str()), a);
}


//		 VEC2

void
PropertyManager::createVec2(wxPropertyGridManager *pg, std::unique_ptr<Attribute> &a) {

	wxPGProperty* topId;

	topId = pg->Append(new wxStringProperty(wxString(a->getName().c_str()), wxPG_LABEL, wxT("<composed>")));

	pg->AppendIn(topId, new wxFloatProperty(wxT("x"), wxPG_LABEL));
	pg->AppendIn(topId, new wxFloatProperty(wxT("y"), wxPG_LABEL));

	if (a->getReadOnlyFlag())
		pg->DisableProperty(topId);
}


void
PropertyManager::updateVec2(wxPropertyGridManager *pg, std::string label, vec2 a) {

	std::string s = label + '.' + "x";
	pg->SetPropertyValue(wxString(s.c_str()), a.x);
	s.clear();
	s = label + '.' + "y";
	pg->SetPropertyValue(wxString(s.c_str()), a.y);
}


//		 VEC3

void
PropertyManager::createVec3(wxPropertyGridManager *pg, std::unique_ptr<Attribute> &a) {

	wxPGProperty* topId;

	topId = pg->Append(new wxStringProperty(wxString(a->getName().c_str()), wxPG_LABEL, wxT("<composed>")));

	pg->AppendIn(topId, new wxFloatProperty(wxT("x"), wxPG_LABEL));
	pg->AppendIn(topId, new wxFloatProperty(wxT("y"), wxPG_LABEL));
	pg->AppendIn(topId, new wxFloatProperty(wxT("z"), wxPG_LABEL));

	if (a->getReadOnlyFlag())
		pg->DisableProperty(topId);
}


void
PropertyManager::updateVec3(wxPropertyGridManager *pg, std::string label, vec3 a) {

	std::string s = label + '.' + "x";
	pg->SetPropertyValue(wxString(s.c_str()), a.x);
	s.clear();
	s = label + '.' + "y";
	pg->SetPropertyValue(wxString(s.c_str()), a.y);
	s.clear();
	s = label + '.' + "z";
	pg->SetPropertyValue(wxString(s.c_str()), a.z);
}


//		 VEC4

void
PropertyManager::createVec4(wxPropertyGridManager *pg, std::unique_ptr<Attribute> &a) {

	wxPGProperty* topId;
	Attribute::Semantics sem = a->getSemantics();

	topId = pg->Append(new wxStringProperty(wxString(a->getName().c_str()), wxPG_LABEL, wxT("<composed>")));

	if (sem == Attribute::Semantics::COLOR) {
		pg->AppendIn(topId, new wxColourProperty(wxT("RGB"), wxPG_LABEL,
			wxColour(255, 255, 255)));
		pg->AppendIn(topId, new wxFloatProperty(wxT("Alpha"), wxPG_LABEL, 1.0));
		pg->Expand(topId);
	}
	else {
		pg->AppendIn(topId, new wxFloatProperty(wxT("x"), wxPG_LABEL));
		pg->AppendIn(topId, new wxFloatProperty(wxT("y"), wxPG_LABEL));
		pg->AppendIn(topId, new wxFloatProperty(wxT("z"), wxPG_LABEL));
		pg->AppendIn(topId, new wxFloatProperty(wxT("w"), wxPG_LABEL));
	}
	if (a->getReadOnlyFlag())
		pg->DisableProperty(topId);
}


void
PropertyManager::updateVec4Color(wxPropertyGridManager *pg, std::string label, vec4 a) {

	wxString rgb = wxString(label.c_str()) + wxString(".RGB");
	wxString alpha = wxString(label.c_str()) + wxString(".Alpha");
	pg->SetPropertyValue(rgb, wxColour(255 * a.x, 255 * a.y, 255 * a.z));
	pg->SetPropertyValue(alpha, a.w);
}


void
PropertyManager::updateVec4(wxPropertyGridManager *pg, std::string label, vec4 a) {

	std::string s = label + '.' + "x";
	pg->SetPropertyValue(wxString(s.c_str()), a.x);
	s.clear();
	s = label + '.' + "y";
	pg->SetPropertyValue(wxString(s.c_str()), a.y);
	s.clear();
	s = label + '.' + "z";
	pg->SetPropertyValue(wxString(s.c_str()), a.z);
	s.clear();
	s = label + '.' + "w";
	pg->SetPropertyValue(wxString(s.c_str()), a.w);
}


//		MAT3

void
PropertyManager::createMat3(wxPropertyGridManager *pg, std::unique_ptr<Attribute> &a) {

	wxPGProperty* topId, *rowId;

	topId = pg->Append(new wxStringProperty(wxString(a->getName().c_str()), wxPG_LABEL, wxT("<composed>")));

	rowId = pg->AppendIn(topId,new wxStringProperty(wxT("Row0"), wxPG_LABEL, wxT("<composed>")));
	pg->AppendIn(rowId, new wxFloatProperty(wxT("x"), wxPG_LABEL));
	pg->AppendIn(rowId, new wxFloatProperty(wxT("y"), wxPG_LABEL));
	pg->AppendIn(rowId, new wxFloatProperty(wxT("z"), wxPG_LABEL));

	rowId = pg->AppendIn(topId, new wxStringProperty(wxT("Row1"), wxPG_LABEL, wxT("<composed>")));
	pg->AppendIn(rowId, new wxFloatProperty(wxT("x"), wxPG_LABEL));
	pg->AppendIn(rowId, new wxFloatProperty(wxT("y"), wxPG_LABEL));
	pg->AppendIn(rowId, new wxFloatProperty(wxT("z"), wxPG_LABEL));

	rowId = pg->AppendIn(topId, new wxStringProperty(wxT("Row2"), wxPG_LABEL, wxT("<composed>")));
	pg->AppendIn(rowId, new wxFloatProperty(wxT("x"), wxPG_LABEL));
	pg->AppendIn(rowId, new wxFloatProperty(wxT("y"), wxPG_LABEL));
	pg->AppendIn(rowId, new wxFloatProperty(wxT("z"), wxPG_LABEL));

	if (a->getReadOnlyFlag())
		pg->DisableProperty(topId);
}


void
PropertyManager::updateMat3(wxPropertyGridManager *pg, std::string label, mat3 a) {

	std::string s = label + ".Row0.x";
	pg->SetPropertyValue(wxString(s.c_str()), a.at(0,0));
	s.clear();
	s = label + ".Row0.y";
	pg->SetPropertyValue(wxString(s.c_str()), a.at(0, 1));
	s.clear();
	s = label + ".Row0.z";
	pg->SetPropertyValue(wxString(s.c_str()), a.at(0, 2));

	s.clear();
	s = label + ".Row1.x";
	pg->SetPropertyValue(wxString(s.c_str()), a.at(1, 0));
	s.clear();
	s = label + ".Row1.y";
	pg->SetPropertyValue(wxString(s.c_str()), a.at(1, 1));
	s.clear();
	s = label + ".Row1.z";
	pg->SetPropertyValue(wxString(s.c_str()), a.at(1, 2));

	s.clear();
	s = label + ".Row2.x";
	pg->SetPropertyValue(wxString(s.c_str()), a.at(2, 0));
	s.clear();
	s = label + ".Row2.y";
	pg->SetPropertyValue(wxString(s.c_str()), a.at(2, 1));
	s.clear();
	s = label + ".Row2.z";
	pg->SetPropertyValue(wxString(s.c_str()), a.at(2, 2));
}


//		MAT4

void
PropertyManager::createMat4(wxPropertyGridManager *pg, std::unique_ptr<Attribute> &a) {

	wxPGProperty* topId, *rowId;

	topId = pg->Append(new wxStringProperty(wxString(a->getName().c_str()), wxPG_LABEL, wxT("<composed>")));

	rowId = pg->AppendIn(topId, new wxStringProperty(wxT("Row0"), wxPG_LABEL, wxT("<composed>")));
	pg->AppendIn(rowId, new wxFloatProperty(wxT("x"), wxPG_LABEL));
	pg->AppendIn(rowId, new wxFloatProperty(wxT("y"), wxPG_LABEL));
	pg->AppendIn(rowId, new wxFloatProperty(wxT("z"), wxPG_LABEL));
	pg->AppendIn(rowId, new wxFloatProperty(wxT("w"), wxPG_LABEL));

	rowId = pg->AppendIn(topId, new wxStringProperty(wxT("Row1"), wxPG_LABEL, wxT("<composed>")));
	pg->AppendIn(rowId, new wxFloatProperty(wxT("x"), wxPG_LABEL));
	pg->AppendIn(rowId, new wxFloatProperty(wxT("y"), wxPG_LABEL));
	pg->AppendIn(rowId, new wxFloatProperty(wxT("z"), wxPG_LABEL));
	pg->AppendIn(rowId, new wxFloatProperty(wxT("w"), wxPG_LABEL));

	rowId = pg->AppendIn(topId, new wxStringProperty(wxT("Row2"), wxPG_LABEL, wxT("<composed>")));
	pg->AppendIn(rowId, new wxFloatProperty(wxT("x"), wxPG_LABEL));
	pg->AppendIn(rowId, new wxFloatProperty(wxT("y"), wxPG_LABEL));
	pg->AppendIn(rowId, new wxFloatProperty(wxT("z"), wxPG_LABEL));
	pg->AppendIn(rowId, new wxFloatProperty(wxT("w"), wxPG_LABEL));

	rowId = pg->AppendIn(topId, new wxStringProperty(wxT("Row3"), wxPG_LABEL, wxT("<composed>")));
	pg->AppendIn(rowId, new wxFloatProperty(wxT("x"), wxPG_LABEL));
	pg->AppendIn(rowId, new wxFloatProperty(wxT("y"), wxPG_LABEL));
	pg->AppendIn(rowId, new wxFloatProperty(wxT("z"), wxPG_LABEL));
	pg->AppendIn(rowId, new wxFloatProperty(wxT("w"), wxPG_LABEL));

	if (a->getReadOnlyFlag())
		pg->DisableProperty(topId);
}


void
PropertyManager::updateMat4(wxPropertyGridManager *pg, std::string label, mat4 a) {

	std::string s = label + ".Row0.x";
	pg->SetPropertyValue(wxString(s.c_str()), a.at(0, 0));
	s.clear();
	s = label + ".Row0.y";
	pg->SetPropertyValue(wxString(s.c_str()), a.at(0, 1));
	s.clear();
	s = label + ".Row0.z";
	pg->SetPropertyValue(wxString(s.c_str()), a.at(0, 2));
	s.clear();
	s = label + ".Row0.w";
	pg->SetPropertyValue(wxString(s.c_str()), a.at(0, 3));

	s.clear();
	s = label + ".Row1.x";
	pg->SetPropertyValue(wxString(s.c_str()), a.at(1, 0));
	s.clear();
	s = label + ".Row1.y";
	pg->SetPropertyValue(wxString(s.c_str()), a.at(1, 1));
	s.clear();
	s = label + ".Row1.z";
	pg->SetPropertyValue(wxString(s.c_str()), a.at(1, 2));
	s.clear();
	s = label + ".Row1.w";
	pg->SetPropertyValue(wxString(s.c_str()), a.at(1, 3));

	s.clear();
	s = label + ".Row2.x";
	pg->SetPropertyValue(wxString(s.c_str()), a.at(2, 0));
	s.clear();
	s = label + ".Row2.y";
	pg->SetPropertyValue(wxString(s.c_str()), a.at(2, 1));
	s.clear();
	s = label + ".Row2.z";
	pg->SetPropertyValue(wxString(s.c_str()), a.at(2, 2));
	s.clear();
	s = label + ".Row2.w";
	pg->SetPropertyValue(wxString(s.c_str()), a.at(2, 3));

	s.clear();
	s = label + ".Row3.x";
	pg->SetPropertyValue(wxString(s.c_str()), a.at(3, 0));
	s.clear();
	s = label + ".Row3.y";
	pg->SetPropertyValue(wxString(s.c_str()), a.at(3, 1));
	s.clear();
	s = label + ".Row3.z";
	pg->SetPropertyValue(wxString(s.c_str()), a.at(3, 2));
	s.clear();
	s = label + ".Row3.w";
	pg->SetPropertyValue(wxString(s.c_str()), a.at(3, 3));
}