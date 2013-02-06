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

#include "enumerate_short_vectors.h"

using namespace std;

void
enumerate_short_vectors
(
 const vector<vector<int>> &qfmatrix,
 unsigned int lower_bound,
 unsigned int upper_bound,
 vector<pair<vector<int>, unsigned int>> &result
 )
{
  size_t m = check_dimensions( qf_matrix );

  if (lower_bound >= upper_bound )
    return;

  mpfr_prec_t current_precision = 53;


  vector<int> res_x;
  int x1;

  vector<vector<mpfr_ptr>> rmatrix;
  vector<mpfr_ptr> rdiag_sqrt;
  // compute all entries with resulting precision current_precision
  auto cholesky = cholesky_decomoposition( qfmatrix, current_precision );
  auto rmatrix = get<0>( cholesky );
  auto rdiag_sqrt = get<1>( cholesky );

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

  auto vec_Ti = vector<mpft_ptr>( m, null_ptr );
  auto vec_Ui = vector<mpft_ptr>( m, null_ptr );
  auto vec_Uij = vector<vector<mpfr_ptr>>( m, vector<mpfr_ptr>( m, null_ptr ) );
  auto vec_Uij_acc vector<vector<mpfr_ptr>>( m, vector<mpfr_ptr>( m, null_ptr ) );

  for ( auto& ptr : vecTi )
    {
      ptr = new mpfr_struct;
      mpft_init2( ptr, current_precision );
    }
  for ( auto& ptr : vecUi )
    {
      ptr = new mpfr_struct;
      mpft_init2( ptr, current_precision );
    }
  for ( auto& row : vec_Uij )
    for ( auto& ptr : row )
      {
	ptr = new mpfr_struct;
	mpft_init2( ptr, current_precision );
      }
  for ( auto& row : vec_Uij_acc )
    for ( auto& ptr : row )
      {
	ptr = new mpfr_struct;
	mpft_init2( ptr, current_precision );
      }


  size_t i{m}, j{m};

  mpfr_ptr C = new mpfr_struct;
  mpfr_ptr Z = new mpfr_struct;
  mpfr_init2( C, current_precision );
  mpfr_init2( Z, current_precision );

  mpfr_ptr LB = new mpfr_struct;
  auto vec_UB = vector<mpfr_ptr>( m, null_ptr );
  mpfr_init2( LB, current_precision );
  for ( auto& ptr : vec_UB )
    {
      ptr = new mpfr_struct;
      mpfr_init2( ptr, current_precision );
    }


  // step 1
  mpfr_set_i( vec_Ti[i], upper_bound, MPFR_RNDU );
  mpfr_set_i( vec_Ui[i], 0, MPFR_RNDZ );
  while ( true )
    {
      init_Z_UB_x( Z, vec_UBi, vec_xi, vec_Ti, rdiag_sqrt, vec_Ui, mpfr_tmp ); // step 2

      while ( true )
	{
	  // step 3
	  mpfr_add_i( vec_x[i], vec_x[i], 1, MPFR_RNDD );

	  for ( size_t j = i + 1; j < m; ++j )
	    mpfr_add( vec_Uij[i][j], vec_Uij[i][j], rmatrix[i][j], rmatrix_round_t[i][j] );

	  if ( mpfr_cmp( vec_x[i], vec_UB[i] ) > 0 ) // step 5
	    {
	      if ( i == 1 ) // goto step 6
		{
		  x1 = mpfr_get_si( vec_x[1], MPFR_RNDNA );
		  if ( x1 == 0 )
		    return;

		  res_x = vector<int>();
		  res_x.push_back( x1 );
		  for ( auto it = ++vec_x.begin(), end = vec_x.cend(); it != end; ++it )
		    res_x.push_back( mpfr_get_si( *it, MPFR_RNDNA ) );

		  
		  // Q(x) = C - T_1 + q_{1, 1} * (x_1 + U_1)^2
		  mpfr_add( mpfr_tmp, vec_xi[1], vec_Ui[1] );
		  mpfr_sqr( mpfr_tmp, mpft_tmp, MPFR_RNDD );
		  mpfr_mul( mpfr_tmp, rmatrix[1][1], mpfr_tmp, MPFR_RNDD );
		  mpfr_sub( mpfr_tmp, vec_Ti[1], mpft_tmp , MPFR_RNDU );

		  result.push_back( tuple<vector<int>, unsigned int>( res_x, upper_bound - mpfr_get_si( mpfr_tmp, MPFR_RNDNA ) ) );
		}
	      else // step 5
		{
		  --i;

		  // U_i = \sum_{j = i + 1}^m q_ij x_j
		  for ( size_t j = i + 1; j < m; ++j )
		    {
		      if ( mpfr_cmp_si( vec_Uij[i][j], 0 ) > 0 )
			mpfr_add( Ui_plus, vec_Uij[i][j], MPFR_RNDD );
		      else
			mpfr_add( Ui_neg, vec_Uij[i][j], MPFR_RNDU );
		    }
		  mpfr_add( vec_Ui[i], Ui_plus, Ui_neg, MPFR_RNDZ );


		  // T_i = T_{i + 1} - q_{i + 1, i + 1} * (x_{i + 1} + U_{i + 1})^2
		  mpfr_add( mpfr_tmp, vec_xi[i + 1], vec_Ui[i + 1] );
		  mpfr_sqr( mpfr_tmp, mpft_tmp, MPFR_RNDD );
		  mpfr_mul( mpfr_tmp, rmatrix[i + 1][i + 1], mpfr_tmp, MPFR_RNDD );
		  mpfr_sub( vec_T[i], vec_T[i + 1], mpft_tmp, MPFR_RNDU );
		  
		  init_Z_UB_x( Z, vec_UBi, vec_xi, vec_Ti, rdiag_sqrt, vec_Ui, mpfr_tmp ); // step 2
		}
	    }
	  else
	    {
	      ++i;
	      continue; // goto step 3
	    }
    }
}

inline
void
init_Z_UB_x
(
 mpfr_prt Z,
 vector<mpfr_ptr>& vec_UBi,
 vector<mpfr_ptr>& vec_xi,
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
  mpfr_sub_i( vec_x[i], LB, 1, MPFR_RNDD );

  // todo: check precisions
  /**
   * We want (UB - LB) / q_ii * \eps < 1
   * and we have to assure that (UB - LB) q_ij has at least 2( i + 1 ) bits after the comma (think about this).
   */
}






