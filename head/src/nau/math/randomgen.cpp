#include <nau/math/randomgen.h>

#include "mtrand.h"

struct nau::math::RandImpl {
  MTRand gen;
};

// Default constructor. Initializes generator with default seed
nau::math::RandomGenerator::RandomGenerator ():
  m_pRandomGen (new RandImpl)
{
}
  
// Destructor
nau::math::RandomGenerator::~RandomGenerator()
{
  delete m_pRandomGen;
}
  
// Get a random integer in the closed interval [MIN,MAX]
int 
nau::math::RandomGenerator::randomInt (int Min, int Max)
{
  unsigned int interval = Max - Min;
  return (Min + m_pRandomGen->gen.randInt (interval));
}

// Overloaded () operator to allow the object to be used as a Random
// Number Generator in STL algorithms. 
// Returns a random integer between 0 and N
  
unsigned int 
nau::math::RandomGenerator::operator () (unsigned int n)
{
  return randomInt (0, n-1);
}
