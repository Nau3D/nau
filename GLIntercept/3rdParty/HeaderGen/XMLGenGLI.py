#!/usr/bin/env python
#-*- coding: ascii -*-

from __future__ import print_function
try:
  from StringIO import StringIO
except:
  from io import StringIO

#  List of extensions that are part of a core version of OpenGL
coreExts = {
	"ARB_vertex_array_object":"VERSION_3_0",
	"ARB_texture_rg":"VERSION_3_0",
	"ARB_texture_compression_rgtc":"VERSION_3_0",
	"ARB_map_buffer_range":"VERSION_3_0",
	"ARB_half_float_vertex":"VERSION_3_0",
	"ARB_framebuffer_sRGB":"VERSION_3_0",
	"ARB_framebuffer_object":"VERSION_3_0",
	"ARB_depth_buffer_float":"VERSION_3_0",
	"ARB_uniform_buffer_object":"VERSION_3_1",
	"ARB_copy_buffer":"VERSION_3_1",
	"ARB_depth_clamp":"VERSION_3_2",
	"ARB_draw_elements_base_vertex":"VERSION_3_2",
	"ARB_fragment_coord_conventions":"VERSION_3_2",
	"ARB_provoking_vertex":"VERSION_3_2",
	"ARB_seamless_cube_map":"VERSION_3_2",
	"ARB_sync":"VERSION_3_2",
	"ARB_texture_multisample":"VERSION_3_2",
	"ARB_vertex_array_bgra":"VERSION_3_2",
	"ARB_texture_rgb10_a2ui":"VERSION_3_3",
	"ARB_texture_swizzle":"VERSION_3_3",
	"ARB_timer_query":"VERSION_3_3",
	"ARB_vertex_type_2_10_10_10_rev":"VERSION_3_3",
	"ARB_blend_func_extended":"VERSION_3_3",
	"ARB_occlusion_query2":"VERSION_3_3",
	"ARB_sampler_objects":"VERSION_3_3",
	"ARB_draw_indirect":"VERSION_4_0",
	"ARB_gpu_shader5":"VERSION_4_0",
	"ARB_gpu_shader_fp64":"VERSION_4_0",
	"ARB_shader_subroutine":"VERSION_4_0",
	"ARB_tessellation_shader":"VERSION_4_0",
	"ARB_transform_feedback2":"VERSION_4_0",
	"ARB_transform_feedback3":"VERSION_4_0",
	"ARB_ES2_compatibility":"VERSION_4_1",
	"ARB_get_program_binary":"VERSION_4_1",
	"ARB_separate_shader_objects":"VERSION_4_1",
	"ARB_vertex_attrib_64bit":"VERSION_4_1",
	"ARB_viewport_array":"VERSION_4_1",
	"ARB_base_instance":"VERSION_4_2",
	"ARB_shading_language_420pack":"VERSION_4_2",
	"ARB_transform_feedback_instanced":"VERSION_4_2",
	"ARB_compressed_texture_pixel_storage":"VERSION_4_2",
	"ARB_conservative_depth":"VERSION_4_2",
	"ARB_internalformat_query":"VERSION_4_2",
	"ARB_map_buffer_alignment":"VERSION_4_2",
	"ARB_shader_atomic_counters":"VERSION_4_2",
	"ARB_shader_image_load_store":"VERSION_4_2",
	"ARB_shading_language_packing":"VERSION_4_2",
	"ARB_texture_storage":"VERSION_4_2",
	"KHR_debug":"VERSION_4_3",
	"ARB_arrays_of_arrays":"VERSION_4_3",
	"ARB_clear_buffer_object":"VERSION_4_3",
	"ARB_compute_shader":"VERSION_4_3",
	"ARB_copy_image":"VERSION_4_3",
	"ARB_ES3_compatibility":"VERSION_4_3",
	"ARB_explicit_uniform_location":"VERSION_4_3",
	"ARB_fragment_layer_viewport":"VERSION_4_3",
	"ARB_framebuffer_no_attachments":"VERSION_4_3",
	"ARB_internalformat_query2":"VERSION_4_3",
	"ARB_invalidate_subdata":"VERSION_4_3",
	"ARB_multi_draw_indirect":"VERSION_4_3",
	"ARB_program_interface_query":"VERSION_4_3",
	"ARB_shader_image_size":"VERSION_4_3",
	"ARB_shader_storage_buffer_object":"VERSION_4_3",
	"ARB_stencil_texturing":"VERSION_4_3",
	"ARB_texture_buffer_range":"VERSION_4_3",
	"ARB_texture_query_levels":"VERSION_4_3",
	"ARB_texture_storage_multisample":"VERSION_4_3",
	"ARB_texture_view":"VERSION_4_3",
	"ARB_vertex_attrib_binding":"VERSION_4_3",
}
  
class Token:
  def __init__(self, name, value):
    assert len(name) > 0
    assert len(value) > 0

    self.name = name
    self.value = value

  def __str__(self):
    return "#define %s %s" % (self.name, self.value)

class Param:
  def __init__(self, type_name, name):
    assert len(type_name) > 0
    assert len(name) > 0

    self.type_name = type_name
    self.name = name

  def __str__(self):
    return "%s %s" % (self.type_name, self.name)

class Function:
  def __init__(self, functionTag, parmTypeLookup):

    self.return_type = parmTypeLookup[functionTag.getAttribute("return")]
    self.name = functionTag.getAttribute("name")

    self.params = []
    for paramTag in functionTag.getElementsByTagName("param"):
      paramName = paramTag.getAttribute("name")
      paramType = parmTypeLookup[paramTag.getAttribute("type")]
      paramKind = paramTag.getAttribute("kind")
      
      if paramKind == "array" or paramKind == "reference":
        paramType += " *";
        paramInput = paramTag.getAttribute("input")
        if paramInput == "true":
          paramType = "const " + paramType
      elif not (paramKind == "value"):
        print("Param Kind %s %s" % (paramName, paramKind))
        assert false
        
      self.params.append(Param(paramType, paramName))
      
  def params_str(self):
    ret = ''
    i = 0
    for i, param in enumerate(self.params):
      ret += str(param)
      if i != len(self.params) - 1:
        ret += ', '
    return ret

  def param_names_str(self):
    ret = ''
    i = 0
    for i, param in enumerate(self.params):
      ret += param.name
      if i != len(self.params) - 1:
        ret += ', '
    return ret

class Extension:
  def __init__(self, name):
    self.name = name
    self.tokens = []
    self.functions = []

def open_ensure_dir(f,flags):
  d = os.path.dirname(f)
  if not os.path.exists(d):
    os.makedirs(d)
  return open(f,flags)

def create_params_str(function):
  ret = ''
  i = 0
  for i, param in enumerate(function.params):
    ret += param.type_name
    if param.type_name == "GLenum":
      ret += "[Main]"
    ret += " "
    ret += param.name
    if i != len(function.params) - 1:
      ret += ', '
  return ret  
  
def create_include_string(prefix,extension):

  header_str = StringIO()
  header_str.write("#define GLI_INCLUDE_%s_%s\n\n" % (prefix.upper(),extension.name.upper()))

  tokens = extension.tokens;
  functions = extension.functions;

  # Abort if no tokens / functions
  if len(tokens) == 0 and len(functions) == 0:
    return ""

  if len(tokens) > 0:
    header_str.write("enum Main {\n\n")
    
    tokenLength  = 0
    for token in tokens:
      if tokenLength < (len(token.name) + 5):
        tokenLength = (len(token.name) + 5)
    
    for token in tokens:
      if token.value[0:2] != "0x" and token.value != "0" and token.value != "1":
        header_str.write("//");
      header_str.write("  %s_%s  = %s,\n" % (prefix.upper(), token.name.ljust(tokenLength), token.value))  
      
    header_str.write("\n};\n\n")    
    
  for function in functions:
    header_str.write("%s %s%s(%s);\n" % (function.return_type, prefix.lower(), function.name, create_params_str(function)))
    
  return header_str.getvalue()
  
  
def create_extension_header(prefix, extensions, base_dir):

  for extension in extensions.items():
    includeFileStr = create_include_string(prefix,extension[1])
    if len(includeFileStr) > 0:
      ext = extension[1].name
      postPrefix = ext[0 : ext.find("_")]
      postPost = ext[ext.find("_")+1 : ]
      
      if prefix.lower() == "gl" :
        if postPrefix.lower() == "version":
          header_file = open_ensure_dir("%s/gli/GLCore/GLCore%s_Include.h" % (base_dir, postPost.lower()), "w")      
        elif postPrefix.upper() == "ARB" or postPrefix.upper() == "ATI" or postPrefix.upper() == "NV" :
          header_file = open_ensure_dir("%s/gli/%s/%s_%s_Include.h" % (base_dir, postPrefix.upper(), postPrefix.upper(), postPost.lower()), "w")
        else:  
          header_file = open_ensure_dir("%s/gli/Misc/%s_%s_Include.h" % (base_dir, postPrefix.upper(), postPost.lower()), "w")      
      elif prefix.lower() == "glx" :
        header_file = open_ensure_dir("%s/gli/XGL/XGL_%s_%s_Include.h" % (base_dir, postPrefix.upper(), postPost.lower()), "w")
      else :
        header_file = open_ensure_dir("%s/gli/%s/%s_%s_%s_Include.h" % (base_dir, prefix.upper(), prefix.upper(), postPrefix.upper(), postPost.lower()), "w")
      
      header_file.write(includeFileStr)
      header_file.close()

      
def parse_xml(prefix,dom,base_dir) :

  extension_set = {}
  
  # Get the  enums and add to each extension  / version 
  enumerationsTag = dom.documentElement.getElementsByTagName("enumerations")
  if (enumerationsTag):
    for enumTag in enumerationsTag[0].getElementsByTagName("enum"):
      
      tokenName = enumTag.getAttribute("name")
      tokenValue = enumTag.getAttribute("value")
      
      # If this is a reference value
      if not tokenValue:
        tokenValue = enumTag.getAttribute("ref")     

      extensionStr = enumTag.getAttribute("version")
      if(extensionStr):
        extensionStr = "VERSION_" + extensionStr.replace(".", "_")
      else:
        # Get access to the extension       
        extensions = enumTag.getElementsByTagName("ext")
        if len(extensions) > 0:
          extensionStr = extensions[0].getAttribute("name")
      
      if (extensionStr) :
        if coreExts.has_key(extensionStr) :
          extensionStr = coreExts[extensionStr]
      
        if extensionStr not in extension_set:
          extension_set[extensionStr] = Extension(extensionStr)
        extension_set[extensionStr].tokens.append(Token(tokenName,tokenValue)) 

  # Get the typemap
  typeMapping = {}
    
  typemapTag = dom.documentElement.getElementsByTagName("typemap")
  if typemapTag :
    for tyepdefTag in typemapTag[0].getElementsByTagName("type-def"):
      typeMapping[tyepdefTag.getAttribute("typename")] = tyepdefTag.getAttribute("C-lang")
  
  #Hack fixup for bad XML data
  typeMapping["void"] = "void"
  
  # Get the functions and add to each extension / version
  functionsTag = dom.documentElement.getElementsByTagName("functions")
  if functionsTag :  
    functionDefsTag = functionsTag[0].getElementsByTagName("function-defs")
    if functionDefsTag :
      for functionTag in functionDefsTag[0].getElementsByTagName("function"):
        extensionStr = functionTag.getAttribute("category").replace("_DEPRECATED", "")
        if coreExts.has_key(extensionStr) :
          extensionStr = coreExts[extensionStr]
        if extensionStr not in extension_set:
          extension_set[extensionStr] = Extension(extensionStr)
        extension_set[extensionStr].functions.append(Function(functionTag, typeMapping)) 
  
  create_extension_header(prefix, extension_set, base_dir)
  
      
def auto_gen_glloader_files(base_dir):

  extension_set = {}

  from xml.dom.minidom import parse

  parse_xml("GL", parse(base_dir + "/opengl.xml"), base_dir) 
  parse_xml("WGL", parse(base_dir + "/wgl.xml"), base_dir)   
  
  #extension_set["GL"].append(Extension(parse(base_dir + "opengl.xml")))
  #extension_set["GLX"].append(Extension(parse(base_dir + "glx.xml")))
  #extension_set["WGL"].append(Extension(parse(base_dir + "wgl.xml")))  
  
  print("")

  #for extensions in extension_set.items():
  #  create_extension_header(extensions[0], extensions[1], base_dir)  

if __name__ == "__main__":
  import os
  import sys
  auto_gen_glloader_files(os.path.dirname(sys.argv[0]))
