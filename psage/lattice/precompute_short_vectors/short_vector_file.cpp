#include "Python.h"

#include <algorithm>
#include <cstdint>
#include <fstream>
#include <tuple>
#include <string>
#include <vector>

#include <iostream>

#include "short_vector_file.h"

using namespace std;


ShortVectorFile::ShortVectorFile(
    PyObject* py_output_file_name,
    PyObject* py_lattice,
    const uint64_t maximal_vector_length
    )
{
    // debug:
    //fstream tst( "tst_file", fstream::out | fstream::binary );
    //long tst_w = 1223124;
    //tst.write( reinterpret_cast<char*>( &tst_w ), sizeof(long) );

    this->read_lattice(py_lattice);

    if ( !PyString_Check( py_output_file_name ) )
    {
        throw( string( "py_output_file_name must be a string" ) );
    }
    string output_file_name = string( PyString_AsString( py_output_file_name ) );
    this->output_file = new fstream( output_file_name,
          fstream::in | fstream::out | fstream::binary );

    this->write_header(maximal_vector_length);
}

ShortVectorFile::ShortVectorFile(
    PyObject* py_output_file_name
    )
{
    string output_file_name = string( PyString_AsString( py_output_file_name ) );
    this->output_file = new fstream( output_file_name,
          fstream::out | fstream::app | fstream::binary );
	  
    this->read_header();
}

ShortVectorFile::~ShortVectorFile()
{
    cout << "called" << endl;
    this->output_file->close();
    delete this->output_file;
    delete this->lattice;
}

void
ShortVectorFile::read_header()
{
    this->read_lattice();
    *this->output_file >> this->maximal_vector_length__cache;
}

void
ShortVectorFile::read_lattice(
    PyObject* py_lattice
    )
{
    Py_ssize_t lattice_rank = PyList_Size( py_lattice );
    this->lattice = new vector<vector<int64_t>>();

    long entry;
    PyObject* py_lattice_row;
    PyObject* py_lattice_entry;
    for ( Py_ssize_t i = 0; i < lattice_rank; ++i )
      {
        py_lattice_row = PyList_GetItem( py_lattice, i );
	if ( lattice_rank != PyList_Size( py_lattice_row ) )
	  throw( string( "Could not read lattice." ) );

	auto current_row = vector<int64_t>();
	for ( Py_ssize_t j = 0; j < lattice_rank; ++j )
	  {
	    py_lattice_entry = PyList_GetItem( py_lattice_row, j );
	    if (! PyInt_Check( py_lattice_entry ) )
              throw( string( "Could not read lattice." ) );

	    current_row.push_back( (int64_t)PyInt_AsLong( py_lattice_entry ) );
	  }
	this->lattice->push_back( current_row );
      }
}

void
ShortVectorFile::read_lattice()
{
    uint64_t lattice_rank;
    this->output_file->seekg( 0, ios_base::beg );
    *this->output_file >> lattice_rank;

    for ( size_t it_row = 0; it_row < lattice_rank; ++it_row )
    {
	vector<int64_t> row = vector<int64_t>();
	this->lattice->push_back( row );
	for ( size_t it_col = 0; it_col < lattice_rank; ++it_col )
        {
	    int64_t entry;
	    *this->output_file >> entry;
	    row.push_back( entry );
	}
    }
}

void
ShortVectorFile::write_header(
    const uint64_t maximal_vector_length
    )
{
    // write the lattice data
    *this << (uint64_t)this->lattice->size();
    for ( auto row_it : *this->lattice )
      for ( auto it : row_it )
        *this << (uint64_t)(it);

    // the maximal lenght of stored vectors
    *this << maximal_vector_length;
    this->maximal_vector_length__cache = maximal_vector_length;

    // entries describing the data that is stored in the file
    for ( uint64_t length = 2; length <= maximal_vector_length; length += 2 ) {
	*this << length << (uint64_t)0 << (uint64_t)0;
    }
}

PyObject*
ShortVectorFile::write_vectors(
    const uint64_t length,
    PyObject* py_vectors
    )
{
    if (! PyList_Check( py_vectors ) )
      return Py_False;

    size_t header_entry_position = (this->lattice->size()^2 + 1)
            * sizeof(int64_t) + (2 * length - 1) * sizeof(int64_t);
    
    this->output_file->seekp( header_entry_position, ios_base::beg );
    *this << (uint64_t)PyList_Size( py_vectors );
    this->output_file->seekg( 0, ios_base::end );
    *this << (uint64_t)( this->output_file->tellg() );

    PyObject* py_vector;
    PyObject* py_entry;
    long entry;

    this->output_file->seekp( 0, ios_base::end );
    try
      {
        for ( Py_ssize_t it = 0; it < PyList_Size( py_vectors ); ++it )
          {
            py_vector = PyList_GetItem( py_vectors, it );
            if (! PyTuple_Check( py_vector ) )
              throw( string( "conversion failed" ) );

            for ( Py_ssize_t it_v = 0; it_v < PyTuple_Size( py_vector ); ++it_v )
              {
                py_entry = PyTuple_GetItem( py_vector, it_v );
                if (! PyInt_Check( py_entry ) )
                  throw( string( "conversion failed" ) );

                entry = PyInt_AsLong( py_entry );
                *this << entry;
              }
        }
      }
    catch ( string &e )
      {
        if ( e == string("conversion failed") )
          {
            this->output_file->seekp( header_entry_position, ios_base::beg );
            *this << (uint64_t)0 << (uint64_t)0;

            return Py_False;
          }
        else
          throw;
      }

    this->output_file->flush();
    *this->output_file << "tst";
    return Py_True;
}

void
ShortVectorFile::increase_maximal_vector_length(
    const uint64_t maximal_vector_length
    )
{
    uint64_t cur_maximal_vector_length = this->maximal_vector_length();
    if ( cur_maximal_vector_length >= maximal_vector_length ) {
	return;
    }

    uint64_t additional_size = 3 * (maximal_vector_length - cur_maximal_vector_length) * sizeof(int64_t);

    vector<pair<size_t, uint64_t>>* cur_blocks = this->data_blocks();
    for ( auto it : *cur_blocks )
      this->move_data_block( it.first, it.first + additional_size, it.second );

    delete cur_blocks;
}

vector<pair<size_t, uint64_t>>*
ShortVectorFile::data_blocks()
{
    vector<pair<size_t, size_t>>* blocks = new vector<pair<size_t, size_t>>;
    uint64_t length, nmb_vectors, position;

    this->output_file->seekg( (this->lattice->size()^2 + 2) * sizeof(int64_t), ios_base::beg );
    for ( size_t it = 0; it < this->maximal_vector_length__cache; ++it )
      {
        *this >> length >> nmb_vectors >> position;
	blocks->push_back( pair<size_t, uint64_t>( position, nmb_vectors ) );
      }
    sort( blocks->begin(), blocks->end() );

    return blocks;
}

void
ShortVectorFile::move_data_block(
    size_t src,
    size_t dest,
    uint64_t nmb_vectors
    )
{
    vector<uint64_t> data;
    uint64_t value;
    
    this->output_file->seekg( src, ios_base::beg );
    for ( size_t it = 0;
	  it < nmb_vectors * this->lattice->size() * sizeof(int64_t);
	  ++it )
      {
	*this >> value;
	data.push_back(value);
      }

    this->output_file->seekp( dest, ios_base::beg );
    for ( auto it : data )
	*this << it;
}

template <class T>
inline
ShortVectorFile&
operator>>(
    ShortVectorFile& stream,
    T& dest
    )
{
    stream.output_file->get( reinterpret_cast<char*>( &dest ), sizeof(T) );
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
