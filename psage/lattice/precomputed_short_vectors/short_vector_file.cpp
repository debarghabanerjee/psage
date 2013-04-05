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

#include <algorithm>
#include <cstdint>
#include <fstream>
#include <map>
#include <string>
#include <tuple>
#include <vector>

#include "short_vector_file.h"

using namespace std;


ShortVectorFile::ShortVectorFile()
{
}

ShortVectorFile::ShortVectorFile
(
 const string& output_file_name,
 const vector<vector<int>>& lattice,
 const unsigned int maximal_vector_length
 )
{
  this->init_with_lattice( output_file_name, lattice, maximal_vector_length );
}


ShortVectorFile::ShortVectorFile
(
 const string& output_file_name
 )
{
  this->init_with_file_name( output_file_name );
}

ShortVectorFile::~ShortVectorFile()
{
  if ( output_file )
    {
      this->output_file->close();
      delete this->output_file;
      this->output_file = nullptr;
    }
}

void
ShortVectorFile::init_with_lattice
(
 const string& output_file_name,
 const vector<vector<int>>& lattice,
 const unsigned int maximal_vector_length
 )
{
  for ( auto& row : lattice )
    this->lattice.push_back( vector<int>( row ) );

  ifstream tmp_istream( output_file_name, fstream::in );
  if ( tmp_istream )
    tmp_istream.close();
  else
    {
      ofstream tmp_ostream( output_file_name, fstream::out );
      tmp_ostream.close();
    }

  this->output_file = new fstream( output_file_name,
                                   fstream::in | fstream::out | fstream::binary );
  if (! this->output_file->is_open() )
    throw( string( "output file must exist already" ) );
  
  this->maximal_vector_length__cache = maximal_vector_length;

  this->next_free_position = this->write_header();
}

void
ShortVectorFile::init_with_file_name
(
 const string& output_file_name
 )
{
  this->output_file = new fstream( output_file_name,
				   fstream::in | fstream::out | fstream::app | fstream::binary );
	  
  this->next_free_position = this->read_header();
}

size_t
ShortVectorFile::read_header()
{
  this->read_lattice();
  uint64_t maximal_vector_length__cache_64;
  *this >> maximal_vector_length__cache_64;
  this->maximal_vector_length__cache = (unsigned int)maximal_vector_length__cache_64;
  this->read_stored_vectors();


  if ( this->stored_vectors__cache.size() == 0 )
    return (1 + this->lattice.size() * this->lattice.size() + 1
	    + 3 * (this->maximal_vector_length__cache << 2) + 1) * sizeof(uint64_t);
  else
    {
      auto last_block = *this->stored_vectors__cache.cend();
      return get<2>( last_block ) + get<1>( last_block ) * this->lattice.size() * sizeof(uint64_t);
    }
}

size_t
ShortVectorFile::write_header()
{
  this->output_file->seekp( 0, ios_base::beg );
  this->write_lattice();
  *this << (uint64_t)this->maximal_vector_length__cache;
  this->write_stored_vectors__empty();

  *this << (uint64_t)0;

  this->output_file->flush();

  return (   1 + this->lattice.size() * this->lattice.size()
           + 1 + 3 * (this->maximal_vector_length__cache / 2) + 1 ) * sizeof(uint64_t);
}


void
ShortVectorFile::read_lattice()
{
  unsigned int lattice_rank;
  uint64_t lattice_rank_64;

  this->output_file->seekg( 0, ios_base::beg );
  *this >> lattice_rank_64;
  lattice_rank = (unsigned int)lattice_rank_64;

  int64_t entry_64;
  for ( size_t it_row = 0; it_row < lattice_rank; ++it_row )
    {
      vector<int> row;
      for ( size_t it_col = 0; it_col < lattice_rank; ++it_col )
        {
	  *this >> entry_64;
	  row.push_back( (int)entry_64 );
	}
      this->lattice.push_back( row );
    }
}

void
ShortVectorFile::write_lattice()
{
  *this << (uint64_t)this->lattice.size();
  for ( auto row_it : this->lattice )
    for ( auto it : row_it )
      *this << (int64_t)it;
}

void
ShortVectorFile::read_stored_vectors()
{
  uint64_t length_64, nmb_vectors_64, position_64;

  for ( unsigned int it = 0; it < this->maximal_vector_length__cache / 2; ++it )
    {
      *this >> length_64 >> nmb_vectors_64 >> position_64;

      if ( position_64 != 0 )
	this->stored_vectors__cache.push_back
	  ( tuple<unsigned int, size_t, size_t>
	    ( (unsigned int)length_64, (size_t)nmb_vectors_64, (size_t)position_64 ) );
    }

  sort( this->stored_vectors__cache.begin(),
	this->stored_vectors__cache.end(),
	[] (tuple<unsigned int, size_t, size_t> fst,
	    tuple<unsigned int, size_t, size_t> snd)
	{ return get<2>( fst ) < get<2>( snd ); } );
}

void
ShortVectorFile::write_stored_vectors__empty()
{
  for ( unsigned int length = 2;
	length <= this->maximal_vector_length__cache;
	length += 2 )
    *this << (uint64_t)length << (uint64_t)0 << (uint64_t)0;
}


vector<vector<int>>
ShortVectorFile::read_vectors
(
 const unsigned int length
 )
{
  size_t nmb_vectors;
  size_t data_position = 0;

  for ( auto& it : this->stored_vectors__cache )
    if ( get<0>( it ) == length )
      {
        nmb_vectors = get<1>( it );
        data_position = get<2>( it );
      }
  if ( data_position == 0 )
    return vector<vector<int>>();

  vector<vector<int>> vectors;
  vector<int> vec;
  int64_t entry_64;
  size_t lattice_rank = this->lattice.size();

  this->output_file->seekg( data_position, ios_base::beg );
  for ( size_t it = 0; it < nmb_vectors; ++it )
    {
      vec = vector<int>();
      for ( size_t it_e = 0; it_e < lattice_rank; ++it_e )
	{
	  *this >> entry_64;
	  vec.push_back( (int)( entry_64 ) );
	}
      vectors.push_back( vec );
    }

  return vectors;
}


bool
ShortVectorFile::write_vectors(
    const unsigned int length,
    vector<vector<int>>& vectors
    )
{
    if ( length > this->maximal_vector_length__cache )
      return false;

    // Verify that this entry has not yet been written.
    for ( auto it : this->stored_vectors__cache )
      if ( get<0>( it ) == length )
        return false;

    // Write the header entry for this set of vectors.
    size_t header_entry_position = (1 + this->lattice.size() * this->lattice.size() + 1 + 3 * (length / 2 - 1) + 1) * sizeof(int64_t);

    this->output_file->seekp( header_entry_position, ios_base::beg );
    *this << (uint64_t)vectors.size();
    this->output_file->seekp( header_entry_position + 8, ios_base::beg );
    *this << (uint64_t)( this->next_free_position );

    int64_t entry_64;

    this->output_file->seekp( this->next_free_position, ios_base::beg );
    for ( auto &vector : vectors )
      for ( auto entry : vector )
        *this << (int64_t)entry;

    this->stored_vectors__cache.push_back
      ( tuple<unsigned int, size_t, size_t>
        ( length, vectors.size(), this->next_free_position ) );
    this->next_free_position +=
      vectors.size() * this->lattice.size() * sizeof(int64_t);

    this->output_file->flush();

    return true;
}

void
ShortVectorFile::increase_maximal_vector_length(
    const unsigned int maximal_vector_length
    )
{
    if ( this->maximal_vector_length__cache >= maximal_vector_length )
      return;

    size_t additional_size =
      3 * ( (maximal_vector_length - this->maximal_vector_length__cache) / 2 ) * sizeof( int64_t );

    for ( auto& it : this->stored_vectors__cache )
      {
        this->move_data_block( get<2>( it ), get<2>( it ) + additional_size, get<1>( it ) );
        get<2>( it ) += additional_size;
      }

    uint64_t nmb_vectors_64, position_64;
    this->output_file->seekg( ( this->lattice.size() * this->lattice.size() + 2 ) * sizeof(int64_t), ios_base::beg );
    this->output_file->seekp( ( this->lattice.size() * this->lattice.size() + 2 ) * sizeof(int64_t), ios_base::beg );
    for ( unsigned int length = 2; length <= this->maximal_vector_length__cache; length += 2 )
      {
        this->output_file->seekg( sizeof(int64_t), ios_base::cur );
        *this >> nmb_vectors_64 >> position_64;
        if ( position_64 != 0 )
          {
            this->output_file->seekp( -2 * sizeof(int64_t), ios_base::cur );
            *this << nmb_vectors_64 << (uint64_t)(position_64 + additional_size);
          }
      }

    for ( unsigned int length = this->maximal_vector_length__cache + 2;
          length <= maximal_vector_length;
          length += 2 )
      *this << (uint64_t)length << (uint64_t)0 << (uint64_t)0;

    *this << (uint64_t)0;

    this->next_free_position += additional_size;

    this->maximal_vector_length__cache = maximal_vector_length;
    this->output_file->seekp( ( this->lattice.size() * this->lattice.size() + 1 ) * sizeof(int64_t) );
    *this << (uint64_t)maximal_vector_length;

    this->output_file->flush();
}

void
ShortVectorFile::move_data_block(
    size_t src,
    size_t dest,
    size_t nmb_vectors
    )
{
    vector<int64_t> data;
    int64_t value;

    this->output_file->seekg( src, ios_base::beg );
    for ( size_t it = 0; it < nmb_vectors * this->lattice.size(); ++it )
      {
	*this >> value;
	data.push_back(value);
      }

    this->output_file->seekp( dest, ios_base::beg );
    for ( auto it : data )
      *this << it;
}

bool
ShortVectorFile::direct_sum
(
 ShortVectorFile &src1,
 ShortVectorFile &src2
 )
{
  size_t dim1 = src1.get_lattice().size();
  size_t dim2 = src2.get_lattice().size();

  if ( dim1 + dim2 != this->lattice.size() )
    return false;

  const vector<vector<int>> &src1_lattice = src1.get_lattice();
  const vector<vector<int>> &src2_lattice = src2.get_lattice();
  for ( size_t row = 0; row < dim1; ++row )
    {
      for ( size_t ind = 0; ind < dim1; ++ind )
	if ( this->lattice[row][ind] != src1_lattice[row][ind] )
	  return false;

      for ( size_t ind = dim1; ind < this->lattice.size(); ++ind )
	if ( this->lattice[row][ind] != 0 )
	  return false;
    }
  for ( size_t row = dim1; row < this->lattice.size(); ++row )
      for ( size_t ind = dim1; ind < this->lattice.size(); ++ind )
	if ( this->lattice[row][ind] != src2_lattice[row - dim1][ind - dim1] )
	  return false;

  map<unsigned int, vector<vector<int>>> src1_vectors;
  map<unsigned int, vector<vector<int>>> src2_vectors;
  
  vector<vector<int>> vectors;
  auto vec = vector<int>( dim1 + dim2, 0 );
  auto nvec = vector<int>( dim1 + dim2, 0 );

  unsigned int maximal_length = min( src1.maximal_vector_length(),
				     src2.maximal_vector_length() );
  if ( maximal_length > this->maximal_vector_length__cache )
    maximal_length = this->maximal_vector_length__cache;


  src1_vectors[0] = vector<vector<int>>();
  src1_vectors[0].emplace_back( vector<int>( this->lattice.size(), 0 ) );
  src2_vectors[0] = vector<vector<int>>();
  src2_vectors[0].emplace_back( vector<int>( this->lattice.size(), 0 ) );
  for ( unsigned int length = 2; length <= maximal_length; length += 2 )
    {
      src1_vectors[length] = src1.read_vectors( length );
      src2_vectors[length] = src2.read_vectors( length );
    }

  for ( unsigned int length = 2; length <= maximal_length; length += 2 )
    {
      vectors = vector<vector<int>>();


      for ( size_t ind = dim1; ind < dim1 + dim2; ++ind )
	vec[ind] = 0;

      for ( auto &vec1 : src1_vectors[length] )
	{
	  for ( size_t ind = 0; ind < dim1; ++ind )
	    vec[ind] = vec1[ind];
	  vectors.push_back( vec );
	}


      for ( size_t ind = 0; ind < dim1; ++ind )
	vec[ind] = 0;

      for ( auto &vec2 : src2_vectors[length] )
	{
	  for ( size_t ind = 0; ind < dim2; ++ind )
	    vec[dim1 + ind] = vec2[ind];
	  vectors.push_back( vec );
	}


      for ( unsigned int sublength = 2; sublength <= length - 2; sublength += 2 )
	{
	  for ( auto &vec1 : src1_vectors[sublength] )
	    {
	      for ( size_t ind = 0; ind < dim1; ++ind )
		{
		  vec[ind] = vec1[ind];
		  nvec[ind] = -vec1[ind];
		}

	      for ( auto &vec2 : src2_vectors[length - sublength] )
		{
		  for ( size_t ind = 0; ind < dim2; ++ind )
		    {
		      vec[dim1 + ind] = vec2[ind];
		      nvec[dim1 + ind] = vec2[ind];
		    }
		  vectors.push_back( vec );
		  vectors.push_back( nvec );
		}
	    }
	}
      this->write_vectors( length, vectors );
    }

  return true;
}
	   

template <class T>
inline
ShortVectorFile&
operator>>(
    ShortVectorFile& stream,
    T& dest
    )
{
    stream.output_file->read( reinterpret_cast<char*>( &dest ), sizeof(T) );
    return stream;
}

template <class T>
inline
ShortVectorFile&
operator<<(
    ShortVectorFile& stream,
    const T& src
    )
{
    stream.output_file->write( reinterpret_cast<const char*>( &src ), sizeof(T) );
    return stream;
}
