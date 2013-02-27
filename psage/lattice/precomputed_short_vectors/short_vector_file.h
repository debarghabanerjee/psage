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

#ifndef __SHORT_VECTOR_FILE_HPP
#define __SHORT_VECTOR_FILE_HPP

#include <fstream>
#include <string>
#include <tuple>
#include <vector>

class ShortVectorFile 
{
public:
  ShortVectorFile();
  ShortVectorFile( const std::string&, const std::vector<std::vector<int>>&, const unsigned int );
  ShortVectorFile( const std::string& );
  ~ShortVectorFile();

  void flush() { this->output_file->flush(); };

  const std::vector<std::vector<int>>& get_lattice() const
  {
    return this->lattice;
  };
 
  unsigned int maximal_vector_length() const
  {
    return this->maximal_vector_length__cache;
  };
  void increase_maximal_vector_length( const unsigned int );

  bool write_vectors( const unsigned int, std::vector<std::vector<int>>& );
  std::vector<std::vector<int>> read_vectors( const unsigned int );

  template <class T> friend inline ShortVectorFile& operator>>( ShortVectorFile&, T& );
  template <class T> friend inline ShortVectorFile& operator<<( ShortVectorFile&, const T& );

  friend class ShortVectorFilePy;

protected:
  void init_with_lattice( const std::string&, const std::vector<std::vector<int>>&, const unsigned int );
  void init_with_file_name( const std::string& );

  std::fstream *output_file = nullptr;
  std::vector<std::vector<int>> lattice;
  // The maximal length that can be stored. The maximum may be attained!
  unsigned int maximal_vector_length__cache;
  std::vector<std::tuple<unsigned int, size_t, size_t>> stored_vectors__cache;
  size_t next_free_position;

private:
  size_t read_header();
  size_t write_header();

  void read_lattice();
  void write_lattice();

  void read_stored_vectors();
  void write_stored_vectors__empty();

  void move_data_block( size_t, size_t, size_t );
};

#endif
