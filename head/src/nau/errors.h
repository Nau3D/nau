#ifndef ERRORS_H
#define ERRORS_H

#include <string>
#include <stdio.h>

//! \brief This file defines the exception handlers in the nau library
//!
//! Exceptions should be used with care and parcimony only to signify serious 
//! errors

#define NAU_THROW(message, ...) \
{\
  char m[1024];\
  sprintf(m, message, ## __VA_ARGS__);\
  throw std::string(m);\
};


namespace nau {
  
  //! \brief The base class for exceptions
  class Exception {
    
  private:
    
    std::string m_Message;
    
  public:
    
    //! Constructor
    Exception(const std::string msg = "Default Exception") : 
      m_Message (msg) {}
    
    //! Destructor
    ~Exception() {}

	 std::string& getException()
	 {
		return m_Message;
	 }
  };
  
  class SceneFactoryError: public Exception {
    
  public:
    
    //! Constructor
    SceneFactoryError(const std::string msg = "Default SceneFactoryError"):
      Exception(msg) {}
  };

  class NauInstanciationError: public Exception {
  
  public:
	//! Constructor
	NauInstanciationError (const std::string msg = "Default NauInstanciationError"):
		Exception (msg) {}
  };

  class ProjectLoaderError : public Exception {
  
  public:
	  //!Constructor
	  ProjectLoaderError (const std::string msg = "Default ProjectLoaderError") : 
		Exception (msg) {}
  };

} 

#endif //ERRORS_H
