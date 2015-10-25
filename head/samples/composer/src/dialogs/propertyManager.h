#ifndef __PROPERTY_MANAGER__
#define __PROPERTY_MANAGER__

// For compilers that support precompilation, includes "wx/wx.h".
#include "wx/wxprec.h"

#ifdef __BORLANDC__
#pragma hdrstop
#endif

#ifndef WX_PRECOMP
#include "wx/wx.h"
#endif

#include <map>
#include <string>

#include "nau/attribute.h"
#include "nau/attributeValues.h"

#include <wx/string.h>
#include <wx/grid.h>
#include <wx/propgrid/propgrid.h>
#include <wx/propgrid/advprops.h>
#include <wx/propgrid/manager.h>



class PropertyManager {

public:
	static void createGrid(wxPropertyGridManager *pg, AttribSet &attribs);
	static void createOrderedGrid(wxPropertyGridManager *pg, AttribSet &attribs, std::vector<std::string> &list);
	static void updateGrid(wxPropertyGridManager *pg, AttribSet &attribs, AttributeValues *attribVal);
	static void updateProp(wxPropertyGridManager *pg, std::string prop, AttribSet &attribs, AttributeValues *attribVal);
	static void setAllReadOnly(wxPropertyGridManager *pg, AttribSet &attribs);
protected:

	static void addAttribute(wxPropertyGridManager *pg, Attribute &a);
	static bool inList(std::string attr, std::vector<std::string> &list);

	static void createEnum(wxPropertyGridManager *pg, Attribute &a);
	static void updateEnum(wxPropertyGridManager *pg, std::string label, int a);

	static void createBool(wxPropertyGridManager *pg, Attribute &a);
	static void updateBool(wxPropertyGridManager *pg, std::string label, bool a);

	static void createBVec4(wxPropertyGridManager *pg, Attribute &a);
	static void updateBVec4(wxPropertyGridManager *pg, std::string label, bvec4 a);

	static void createInt(wxPropertyGridManager *pg, Attribute &a);
	static void updateInt(wxPropertyGridManager *pg, std::string label, int a);

	static void createIVec3(wxPropertyGridManager *pg, Attribute &a);
	static void updateIVec3(wxPropertyGridManager *pg, std::string label, ivec3 a);

	static void createUInt(wxPropertyGridManager *pg, Attribute &a);
	static void updateUInt(wxPropertyGridManager *pg, std::string label, unsigned int a);

	static void createUIVec2(wxPropertyGridManager *pg, Attribute &a);
	static void updateUIVec2(wxPropertyGridManager *pg, std::string label, uivec2 a);

	static void createUIVec3(wxPropertyGridManager *pg, Attribute &a);
	static void updateUIVec3(wxPropertyGridManager *pg, std::string label, uivec3 a);

	static void createFloat(wxPropertyGridManager *pg, Attribute &a);
	static void updateFloat(wxPropertyGridManager *pg, std::string label, float a);

	static void createVec2(wxPropertyGridManager *pg, Attribute &a);
	static void updateVec2(wxPropertyGridManager *pg, std::string label, vec2 a);

	static void createVec3(wxPropertyGridManager *pg, Attribute &a);
	static void updateVec3(wxPropertyGridManager *pg, std::string label, vec3 a);

	static void createVec4(wxPropertyGridManager *pg, Attribute &a);
	static void updateVec4(wxPropertyGridManager *pg, std::string label, vec4 a);
	static void updateVec4Color(wxPropertyGridManager *pg, std::string label, vec4 a);

	static void createMat3(wxPropertyGridManager *pg, Attribute &a);
	static void updateMat3(wxPropertyGridManager *pg, std::string label, mat3 a);

	static void createMat4(wxPropertyGridManager *pg, Attribute &a);
	static void updateMat4(wxPropertyGridManager *pg, std::string label, mat4 a);

};

#endif