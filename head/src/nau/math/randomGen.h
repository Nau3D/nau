#ifndef RANDOMGEN_H
#define RANDOMGEN_H

/**
 * \file   randomGen.h
 * \author  <pangelo@doublemv.com>
 * \date   Sat Jun 28 19:01:25 2008
 * 
 * \brief This class abstracts random number generation 
 * 
 * Internally it uses the open source mersenne twister generator and
 * provides an overloaded () operator to allow using this class as a
 * RNG in STL containers and algorithms
 */

namespace nau {

  namespace math {

    class RandomGenerator {

    private:

      // Forward declare internal implementation
      struct RandImpl *m_pRandomGen;

    public:
  
      // Default constructor. Initializes generator with default seed
      RandomGenerator ();
  
      // Destructor
      ~RandomGenerator();
  
      // Get a random integer in the closed interval [MIN,MAX]
      int randomInt (int Min, int Max);

      // Overloaded () operator to allow the object to be used as a Random
      // Number Generator in STL algorithms. 
      // Returns a random integer between 0 and N
      unsigned int operator () (unsigned int n);
    };

  } // namespace math

} // namespace nau

#endif // RANDOMGEN_H
