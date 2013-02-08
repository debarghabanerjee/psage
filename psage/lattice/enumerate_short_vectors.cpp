/**
 *
 * Copyright (C) 2013 Martin Raum
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

/**
 * Implement Fincke-Pohst algorithm with multi precision floating points.
 */

#include "limits.h"
#include <utility>
#include <vector>

#include <iostream>

#include "enumerate_short_vectors.h"

using namespace std;


tuple< vector<vector<mpfr_ptr>>, vector<mpfr_ptr> >
cholesky_decomposition
(
 const vector<vector<int>>& qfmatrix,
 mpfr_prec_t final_precision
 )
{
  // todo: consider precisions
  mpfr_ptr mpfr_tmp = new __mpfr_struct;
  mpfr_init2( mpfr_tmp, final_precision );


  size_t m = qfmatrix.size();
  vector<vector<mpfr_ptr>> rmatrix;
  vector<mpfr_ptr> row_mpfr;

  for ( auto &row : qfmatrix )
    {
      row_mpfr = vector<mpfr_ptr>();
      for ( size_t i = 0; i < m; ++i )
  	{
  	  mpfr_ptr tmp = new __mpfr_struct;
  	  mpfr_init2( tmp, final_precision );
  	  row_mpfr.push_back( tmp );
  	}

      for ( size_t ind = 0; ind < m; ++ind )
  	mpfr_set_si( row_mpfr[ind], row[ind], MPFR_RNDZ );

      rmatrix.push_back( row_mpfr );
    }
    
  for ( size_t i = 0; i < m; ++i )
    for ( size_t j = i + 1; j < m; ++j )
      {
  	mpfr_set( rmatrix[j][i], rmatrix[i][j], MPFR_RNDZ );
  	mpfr_div( rmatrix[i][j], rmatrix[i][j], rmatrix[i][i], MPFR_RNDZ );
      }


  for ( size_t i = 0; i < m; ++i )
    for ( size_t k = i + 1; k < m; ++k )
      for ( size_t l = k; l < m; ++l )
  	{
  	  mpfr_mul( mpfr_tmp, rmatrix[k][i], rmatrix[i][l], MPFR_RNDD );
  	  mpfr_sub( rmatrix[k][l], rmatrix[k][l], mpfr_tmp, MPFR_RNDU );
  	}


  mpfr_clear( mpfr_tmp );
  delete mpfr_tmp;

  
  vector<mpfr_ptr> rdiag_sqrt;
  for ( size_t i = 0; i < m; ++i )
    {
      mpfr_ptr tmp = new __mpfr_struct;
      mpfr_init2( tmp, final_precision );
      rdiag_sqrt.push_back( tmp );
    }

  for ( size_t i = 0; i < m; ++i )
    mpfr_sqrt( rdiag_sqrt[i], rmatrix[i][i], MPFR_RNDZ );

  return tuple< vector<vector<mpfr_ptr>>, vector<mpfr_ptr> >( rmatrix, rdiag_sqrt );
}


void
enumerate_short_vectors
(
 const vector<vector<int>> &qfmatrix,
 unsigned int lower_bound,
 unsigned int upper_bound,
 vector<pair<vector<int>, unsigned int>> &result
 )
{
  size_t m = qfmatrix.size();
  for ( auto &row : qfmatrix )
    if ( row.size() != m )
      return;

  if (lower_bound >= upper_bound )
    return;

  mpfr_prec_t current_precision = 53;

  vector<int> res_x;
  int x1;

  // compute all entries with resulting precision current_precision
  auto cholesky = cholesky_decomposition( qfmatrix, current_precision );
  auto rmatrix = get<0>( cholesky );
  auto rdiag_sqrt = get<1>( cholesky );

  cout << "compute cholesky" << endl;
  for ( auto &row : rmatrix )
    {
      for ( auto e : row )
	{
	  cout << mpfr_get_d1( e ) << " ";
	}
      cout << endl;
    }
  cout << endl;
    
  mpfr_ptr mpfr_tmp = new __mpfr_struct;
  mpfr_init2( mpfr_tmp, current_precision );

  vector<vector<mpfr_rnd_t>> rmatrix_round_t;
  for ( auto& row : rmatrix )
    {
      vector<mpfr_rnd_t> row_round_t;
      for ( auto ptr : row )
  	{
  	  if ( mpfr_cmp_si( ptr, 0 ) > 0 )
  	    row_round_t.push_back( MPFR_RNDD );
  	  else
  	    row_round_t.push_back( MPFR_RNDU );
  	}
      rmatrix_round_t.push_back( row_round_t );
    }

  vector<mpfr_ptr> vec_x;
  for ( size_t i = 0; i < m; ++i )
    {
      mpfr_ptr tmp = new __mpfr_struct;
      mpfr_init2( tmp, current_precision );
      vec_x.push_back( tmp );
    }

  vector<mpfr_ptr> vec_Ti;
  vector<mpfr_ptr> vec_Ui;
  vector<vector<mpfr_ptr>> vec_Uij;
  vector<vector<mpfr_ptr>> vec_Uij_acc;

  for ( size_t i = 0; i < m; ++i )
    {
      mpfr_ptr tmp = new __mpfr_struct;
      mpfr_init2( tmp, current_precision );
      vec_Ti.push_back( tmp );
    }
  for ( size_t i = 0; i < m; ++i )
    {
      mpfr_ptr tmp = new __mpfr_struct;
      mpfr_init2( tmp, current_precision );
      vec_Ui.push_back( tmp );
    }
  for ( size_t i = 0; i < m; ++i )
    {
      vector<mpfr_ptr> row;
      for ( size_t j = 0; j < m; ++j )
  	{
  	  mpfr_ptr tmp = new __mpfr_struct;
  	  mpfr_init2( tmp, current_precision );
  	  row.push_back( tmp );
  	}
      vec_Uij.push_back( row );
    }
  for ( size_t i = 0; i < m; ++i )
    {
      vector<mpfr_ptr> row;
      for ( size_t j = 0; j < m; ++j )
  	{
  	  mpfr_ptr tmp = new __mpfr_struct;
  	  mpfr_init2( tmp, current_precision );
  	  row.push_back( tmp );
  	}
      vec_Uij_acc.push_back( row );
    }

  mpfr_ptr Ui_plus = new __mpfr_struct;
  mpfr_ptr Ui_neg = new __mpfr_struct;
  mpfr_init2( Ui_plus, current_precision );
  mpfr_init2( Ui_neg, current_precision );

  mpfr_ptr C = new __mpfr_struct;
  mpfr_ptr Z = new __mpfr_struct;
  mpfr_init2( C, current_precision );
  mpfr_init2( Z, current_precision );

  mpfr_ptr LB = new __mpfr_struct;
  mpfr_init2( LB, current_precision );
  vector<mpfr_ptr> vec_UB;
  for ( size_t i = 0; i < m; ++i )
    {
      mpfr_ptr tmp = new __mpfr_struct;
      mpfr_init2( tmp, current_precision );
      vec_UB.push_back( tmp );
    }

  size_t i{ m }, j{ m };

  cout << "initialized everything" << endl;


  cout << "i " << i << endl;
  cout << vec_Ti.size() << " " << vec_Ti[i] << endl;
  cout << vec_Ti[i]->_mpfr_prec << " " << vec_Ti[i]->_mpfr_sign << " " << vec_Ti[i]->_mpfr_exp << vec_Ti[i]->_mpfr_d << endl;
  cout << upper_bound << endl;
  // step 1
  mpfr_set_ui( vec_Ti[i], upper_bound, MPFR_RNDU );
  cout << "set Ti" << endl;
  mpfr_set_ui( vec_Ui[i], 0, MPFR_RNDZ );

  cout << "set Ti, Ui" << endl;

  while ( true )
    {
      init_Z_UB_x( i, Z, LB, vec_UB, vec_x, vec_Ti, rdiag_sqrt, vec_Ui, mpfr_tmp ); // step 2

      while ( true )
  	{
  	  // step 3
  	  mpfr_add_si( vec_x[i], vec_x[i], 1, MPFR_RNDD );

  	  for ( size_t j = i + 1; j < m; ++j )
  	    mpfr_add( vec_Uij[i][j], vec_Uij[i][j], rmatrix[i][j], rmatrix_round_t[i][j] );

  	  if ( mpfr_cmp( vec_x[i], vec_UB[i] ) > 0 ) // step 5
  	    {
  	      if ( i == 1 ) // goto step 6
  		{
  		  x1 = mpfr_get_si( vec_x[1], MPFR_RNDNA );
  		  if ( x1 == 0 )
  		    // todo: is this right or should we check all entries
  		    break;

  		  res_x = vector<int>();
  		  res_x.push_back( x1 );
  		  for ( auto it = ++vec_x.begin(), end = vec_x.end(); it != end; ++it )
  		    res_x.push_back( mpfr_get_si( *it, MPFR_RNDNA ) );

		  
  		  // Q(x) = C - T_1 + q_{1, 1} * (x_1 + U_1)^2
  		  mpfr_add( mpfr_tmp, vec_x[1], vec_Ui[1], MPFR_RNDD );
  		  mpfr_sqr( mpfr_tmp, mpfr_tmp, MPFR_RNDD );
  		  mpfr_mul( mpfr_tmp, rmatrix[1][1], mpfr_tmp, MPFR_RNDD );
  		  mpfr_sub( mpfr_tmp, vec_Ti[1], mpfr_tmp , MPFR_RNDU );

  		  result.push_back( pair<vector<int>, unsigned int>( res_x, upper_bound - mpfr_get_si( mpfr_tmp, MPFR_RNDNA ) ) );
  		}
  	      else // step 5
  		{
  		  --i;

  		  mpfr_set_si( Ui_plus, 0, MPFR_RNDZ );
  		  mpfr_set_si( Ui_neg, 0, MPFR_RNDZ );

  		  // U_i = \sum_{j = i + 1}^m q_ij x_j
  		  for ( size_t j = i + 1; j < m; ++j )
  		    {
  		      if ( mpfr_cmp_si( vec_Uij[i][j], 0 ) > 0 )
  			mpfr_add( Ui_plus, Ui_plus, vec_Uij[i][j], MPFR_RNDD );
  		      else
  			mpfr_add( Ui_neg, Ui_neg, vec_Uij[i][j], MPFR_RNDU );
  		    }
  		  mpfr_add( vec_Ui[i], Ui_plus, Ui_neg, MPFR_RNDZ );


  		  // T_i = T_{i + 1} - q_{i + 1, i + 1} * (x_{i + 1} + U_{i + 1})^2
  		  mpfr_add( mpfr_tmp, vec_x[i + 1], vec_Ui[i + 1], MPFR_RNDD );
  		  mpfr_sqr( mpfr_tmp, mpfr_tmp, MPFR_RNDD );
  		  mpfr_mul( mpfr_tmp, rmatrix[i + 1][i + 1], mpfr_tmp, MPFR_RNDD );
  		  mpfr_sub( vec_Ti[i], vec_Ti[i + 1], mpfr_tmp, MPFR_RNDU );
		  
  		  init_Z_UB_x( i, Z, LB, vec_UB, vec_x, vec_Ti, rdiag_sqrt, vec_Ui, mpfr_tmp ); // step 2
  		}
  	    }
  	  else
  	    {
  	      ++i;
  	      continue; // goto step 3
  	    }
  	}
    }


  for ( auto &row : rmatrix )
    for ( auto ptr : row )
      mpfr_clear( ptr );
  for ( auto ptr : rdiag_sqrt )
    mpfr_clear( ptr );

  mpfr_clear( mpfr_tmp );

  for ( auto ptr : vec_Ti )
    mpfr_clear( ptr );
  for ( auto ptr : vec_Ui )
    mpfr_clear( ptr );
  for ( auto &row : vec_Uij )
    for ( auto ptr : row )
      mpfr_clear( ptr );
  for ( auto &row : vec_Uij_acc )
    for ( auto ptr : row )
      mpfr_clear( ptr );

  mpfr_clear( Ui_plus );
  mpfr_clear( Ui_neg );

  mpfr_clear( C );
  mpfr_clear( Z );
  
  mpfr_clear( LB );
  for ( auto ptr : vec_UB )
    mpfr_clear( ptr );
}

inline
void
init_Z_UB_x
(
 size_t i,
 mpfr_ptr Z,
 mpfr_ptr LB,
 vector<mpfr_ptr>& vec_UB,
 vector<mpfr_ptr>& vec_x,
 vector<mpfr_ptr>& vec_Ti,
 vector<mpfr_ptr>& rdiag_sqrt,
 vector<mpfr_ptr>& vec_Ui,
 mpfr_ptr mpfr_tmp
 )
{
  // Z = (T_i / q_ii)^(1/2)
  mpfr_sqrt( mpfr_tmp, vec_Ti[i], MPFR_RNDD );
  mpfr_div( Z, mpfr_tmp, rdiag_sqrt[i], MPFR_RNDD );

  // UB_i = floor( Z - U_i )
  mpfr_sub( vec_UB[i], Z, vec_Ui[i], MPFR_RNDD );
  // x_i = ceil( - Z - U_i ) - 1
  mpfr_add( mpfr_tmp, Z, vec_Ui[i], MPFR_RNDD );
  mpfr_neg( LB, mpfr_tmp, MPFR_RNDU );
  mpfr_sub_si( vec_x[i], LB, 1, MPFR_RNDD );

  // todo: check precisions
  /**
   * We want (UB - LB) / q_ii * \eps < 1
   * and we have to assure that (UB - LB) q_ij has at least 2( i + 1 ) bits after the comma (think about this).
   */
}
