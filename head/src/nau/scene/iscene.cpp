#include <nau/scene/iscene.h>

#include <nau.h>
#include <nau/math/simpletransform.h>

using namespace nau::math;
using namespace nau::scene;

bool 
IScene::Init() {

	// VEC4
	Attribs.add(Attribute(SCALE, "SCALE", Enums::DataType::VEC4, false, new vec4(1.0f, 1.0f, 1.0f, 1.0f)));
	Attribs.add(Attribute(TRANSLATE, "TRANSLATE", Enums::DataType::VEC4, false, new vec4(0.0f, 0.0f, -1.0f, 0.0f)));
	Attribs.add(Attribute(ROTATE, "ROTATE", Enums::DataType::VEC4, false, new vec4(0.0f, 0.0f, 1.0f, 0.0f)));

	NAU->registerAttributes("SCENE", &Attribs);

	return true;
}

AttribSet IScene::Attribs;
bool IScene::Inited = Init();

void
IScene::setPropf4(Float4Property prop, vec4& aVec) {

	ITransform *tis = TransformFactory::create("SimpleTransform");

	switch (prop) {
	case SCALE:
		tis->scale(aVec.x, aVec.y, aVec.z);
		transform(tis);
		break;
	case ROTATE:
		tis->rotate(aVec.w, aVec.x, aVec.y, aVec.z);
		transform(tis);
		break;
	case TRANSLATE:
		tis->translate(aVec.x, aVec.y, aVec.z);
		transform(tis);
		break;
	}
	m_Float4Props[prop] = aVec;

}