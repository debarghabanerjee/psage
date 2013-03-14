/**
 *
 * Copyright (C) 2012 Martin Raum
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU General Public License
 * as published by the Free Software Foundation; either version 3
 * of the License, or (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
 * General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, see <http://www.gnu.org/licenses/>.
 *
 */

#ifndef SHORTVECTORFILEPY_H_
#define SHORTVECTORFILEPY_H_

#include "Python.h"

#include <fstream>
#include <string>
#include <tuple>
#include <vector>


#include "short_vector_file.h"

class ShortVectorFilePy : public ShortVectorFile
{
public:
  ShortVectorFilePy( PyObject*, PyObject*, const unsigned int );
  ShortVectorFilePy( PyObject* );
  virtual ~ShortVectorFilePy();

  PyObject* get_lattice_py() const;

  PyObject* stored_vectors_py();
  PyObject* write_vectors_py( const unsigned int, PyObject* );
  PyObject* read_vectors_py( const unsigned int );

  template <class T> friend inline ShortVectorFilePy& operator>>( ShortVectorFilePy&, T& );
  template <class T> friend inline ShortVectorFilePy& operator<<( ShortVectorFilePy&, const T& );

private:
  std::vector<std::vector<int>> parse_python_lattice( PyObject* );
};

#endif /* SHORTVECTORFILEPY_H_ */
