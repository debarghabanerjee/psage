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
#include "enumerate_short_vectors_internal.h"

using namespace std;

static const size_t precision_increment = 3;

void
cholesky_decomposition
(
 const vector<vector<int>>& qfmatrix,
 vector<vector<mpfi_ptr>> rmatrix,
 vector<mpfi_ptr> rdiag_sqrt,
 mpfi_ptr mpfi_tmp
 )
{
  size_t m = qfmatrix.size();

  for ( size_t i = 0; i < m; ++i )
    for ( size_t j = 0; j < m; ++j )
      mpfi_set_si( rmatrix[i][j], qfmatrix[i][j] );

  // step 1 in Fincke-Pohst
  // q_ji <- q_ij, q_ij <- q_ij / q_ii
    
  for ( size_t i = 0; i < m; ++i )
    for ( size_t j = i + 1; j < m; ++j )
      {
  	mpfi_set( rmatrix[j][i], rmatrix[i][j] );
  	mpfi_div( rmatrix[i][j], rmatrix[i][j], rmatrix[i][i] );
      }

  // step 2 in Fincke-Pohst
  // q_kl <- q_kl - q_ki q_il

  for ( size_t i = 0; i < m; ++i )
    for ( size_t k = i + 1; k < m; ++k )
      for ( size_t l = k; l < m; ++l )
  	{
  	  mpfi_mul( mpfi_tmp, rmatrix[k][i], rmatrix[i][l] );
  	  mpfi_sub( rmatrix[k][l], rmatrix[k][l], mpfi_tmp );
  	}

  // we later need \sqrt{q_ii}, which we precompute here.
  for ( size_t i = 0; i < m; ++i )
    mpfi_sqrt( rdiag_sqrt[i], rmatrix[i][i] );
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
  // check dimensions of the quadratic form matrix
  size_t m = qfmatrix.size();
  for ( auto &row : qfmatrix )
    if ( row.size() != m )
      return;

  if (lower_bound >= upper_bound )
    return;

  // standard precision
  mp_prec_t precision = 53;


  // create varibles
  mpfr_ptr mpfr_tmp = new __mpfr_struct;
  mpfi_ptr mpfi_tmp = new __mpfi_struct;
  mpfi_ptr mpfi_tmp2 = new __mpfi_struct;

  vector<vector<mpfi_ptr>> rmatrix;
  vector<mpfi_ptr> rdiag_sqrt;

  vector<mpfi_ptr> vec_Ti;
  vector<mpfi_ptr> vec_Ui;
  vector<vector<mpfi_ptr>> vec_Uij;

  mpfi_ptr C = new __mpfi_struct;
  mpfi_ptr Z = new __mpfi_struct;

  int LB;
  auto vec_UB = vector<int>(m, 0);
  int IB_lower, IB_upper;

  vector<int> vec_x = vector<int>(m, 0);
  int int_tmp;
  bool x_is_zero;

  // we use zero based indices, so we start with i = m - 1
  size_t i{ m - 1 };

  init( m, vec_Ti, vec_Ui, vec_Uij, C, Z, rmatrix, rdiag_sqrt, mpfr_tmp, mpfi_tmp, mpfi_tmp2 );

  recompute( m - 1, m, upper_bound, vec_x, LB, vec_UB, vec_Ti, vec_Ui, vec_Uij, C, Z,
	     qfmatrix, rmatrix, rdiag_sqrt, mpfi_tmp, mpfr_tmp, precision );
  vec_x[m - 1] = LB;
  for ( size_t j = 0; j < m - 1; ++j )
    mpfi_mul_si( vec_Uij[j][m - 1], rmatrix[j][m - 1], LB );

  while ( true )
    {
      // step 3
      ++vec_x[i];

      // in order to implement the lower bound, we have this extra condition
      if ( i == 0 && vec_x[0] == IB_lower )
	{
	  vec_x[0] = IB_upper;
	  mpfi_mul_si( vec_Uij[0][0], rmatrix[0][0], IB_upper);
	}
      else
	for ( size_t j = 0; j <= i; ++j )
	  mpfi_add( vec_Uij[j][i], vec_Uij[j][i], rmatrix[j][i] );

      if ( vec_x[i] > vec_UB[i] ) // goto step 5
        {
          ++i;
          continue; // goto step 3
        }
      else
        {
          if ( i == 0 ) // goto step 6
            {
              x_is_zero = true;
	      for ( auto e : vec_x )
		if ( e != 0 )
		  {
		    x_is_zero = false;
		    break;
		  }
              if ( x_is_zero )
                break;

              // Q(x) = C - T_1 + q_{1, 1} * (x_1 + U_1)^2
              mpfi_add_si( mpfi_tmp, vec_Ui[0], vec_x[0] );
              mpfi_sqr( mpfi_tmp, mpfi_tmp );
              mpfi_mul( mpfi_tmp, rmatrix[0][0], mpfi_tmp );
              mpfi_sub( mpfi_tmp, vec_Ti[0], mpfi_tmp );

	      if ( !mpfi_get_unique_si( int_tmp, mpfi_tmp, mpfr_tmp ) )
		{
		  if ( vec_x[0] == IB_upper )
		    vec_x[0] = IB_lower - 1;
		  else
		    --vec_x[0];

		  recompute( 0, m, upper_bound, vec_x, LB, vec_UB, vec_Ti, vec_Ui, vec_Uij, C, Z,
			     qfmatrix, rmatrix, rdiag_sqrt, mpfi_tmp, mpfr_tmp, precision );
		  continue;
		}
	      
	      result.push_back( pair<vector<int>, unsigned int>( vec_x, upper_bound - int_tmp ) );
            }
          else // step 5
            {
              --i;

              // U_i = \sum_{j = i + 1}^m q_ij x_j
              mpfi_set_si( vec_Ui[i], 0 );
              for ( size_t j = i + 1; j < m; ++j )
		mpfi_add( vec_Ui[i], vec_Ui[i], vec_Uij[i][j] );

              // T_i = T_{i + 1} - q_{i + 1, i + 1} * (x_{i + 1} + U_{i + 1})^2
              mpfi_add_si( mpfi_tmp, vec_Ui[i + 1], vec_x[i + 1] );
              mpfi_sqr( mpfi_tmp, mpfi_tmp );
              mpfi_mul( mpfi_tmp, rmatrix[i + 1][i + 1], mpfi_tmp );
              mpfi_sub( vec_Ti[i], vec_Ti[i + 1], mpfi_tmp );

	      // step 2
	      if ( !step_2( i, vec_x, LB, vec_UB, Z, vec_Ti, vec_Ui, vec_Uij, rmatrix, rdiag_sqrt, mpfi_tmp, mpfr_tmp, true ) )
		{
		  ++i;
		  --vec_x[i];
		  recompute( i, m, upper_bound, vec_x, LB, vec_UB, vec_Ti, vec_Ui, vec_Uij, C, Z,
			     qfmatrix, rmatrix, rdiag_sqrt, mpfi_tmp, mpfr_tmp, precision );
		  continue;
		}
	      else
		vec_x[i] = LB;

	      // compute intermediate bounds corresponding to lower_bound
	      if ( i == 0 )
		{
		  mpfi_sub_ui( mpfi_tmp, vec_Ti[0], upper_bound - lower_bound );

		  if ( mpfr_cmp_si( &mpfi_tmp->right, 0 ) < 0 )
		    {
		      // independent of vec_x[0], the lower bound will never be attained.
		      IB_lower = vec_x[i];
		      continue;
		    }
		  else if ( mpfr_cmp_si( &mpfi_tmp->left, 0 ) < 0 )
		    // adjust the bound, so that we can compute the square root
		    mpfr_set_si( &mpfi_tmp->left, 0, MPFR_RNDZ );

		  mpfi_sqrt( mpfi_tmp, mpfi_tmp );
		  mpfi_div( mpfi_tmp, mpfi_tmp, rdiag_sqrt[0] );

		  mpfi_get_unique_floor_si( IB_lower, mpfi_tmp, mpfr_tmp );

		  mpfi_set( mpfi_tmp2, mpfi_tmp );
		  
		  mpfi_add( mpfi_tmp, mpfi_tmp, vec_Ui[0] );
		  mpfi_get_unique_floor_si( IB_lower, mpfi_tmp, mpfr_tmp );

		  mpfi_neg( mpfi_tmp, mpfi_tmp );
		  mpfi_add_si( mpfi_tmp, mpfi_tmp, 1 );
		  mpfi_get_unique_floor_si( IB_lower, mpfi_tmp, mpfr_tmp );

		  if ( !mpfi_get_unique_floor_si( IB_lower, mpfi_tmp, mpfr_tmp ) )
		    {
		      ++i;
		      --vec_x[i];
		      recompute( i, m, upper_bound, vec_x, LB, vec_UB, vec_Ti, vec_Ui, vec_Uij, C, Z,
				 qfmatrix, rmatrix, rdiag_sqrt, mpfi_tmp, mpfr_tmp, precision );
		      continue;
		    }

		  mpfi_sub( mpfi_tmp, mpfi_tmp2, vec_Ui[0] );
		  if ( !mpfi_get_unique_ceil_si( IB_upper, mpfi_tmp, mpfr_tmp ) )
		    {
		      ++i;
		      --vec_x[i];
		      recompute( i, m, upper_bound, vec_x, LB, vec_UB, vec_Ti, vec_Ui, vec_Uij, C, Z,
				 qfmatrix, rmatrix, rdiag_sqrt, mpfi_tmp, mpfr_tmp, precision );
		      continue;
		    }

		  // We must not prevent the algorithm from terminating.  If
		  // all but the first entry vanish, we therefore set
		  // IB_upper to 0.
		  x_is_zero = true;
		  for ( auto it = vec_x.begin() + 1, it_end = vec_x.end();
			it != it_end; ++it )
		    if ( *it != 0 )
		      {
			x_is_zero = false;
			break;
		      }
		  if ( x_is_zero )
		    IB_upper = 0;
		}
            }
        }
    }

  // clear variables
  clear( vec_Ti, vec_Ui, vec_Uij, C, Z, rmatrix, rdiag_sqrt, mpfr_tmp, mpfi_tmp, mpfi_tmp2 );
}

inline void
init
(
 size_t m,
 vector<mpfi_ptr> &vec_Ti,
 vector<mpfi_ptr> &vec_Ui,
 vector<vector<mpfi_ptr>> &vec_Uij,
 mpfi_ptr &C,
 mpfi_ptr &Z,
 vector<vector<mpfi_ptr>> &rmatrix,
 vector<mpfi_ptr> &rdiag_sqrt,
 mpfr_ptr &mpfr_tmp,
 mpfi_ptr &mpfi_tmp,
 mpfi_ptr &mpfi_tmp2
 )
{
  mpfr_init( mpfr_tmp );
  mpfi_init( mpfi_tmp );
  mpfi_init( mpfi_tmp2 );

  mpfi_init_matrix( rmatrix, m );
  mpfi_init_vector( rdiag_sqrt, m );

  mpfi_init_vector( vec_Ti, m );
  mpfi_init_vector( vec_Ui, m );
  mpfi_init_matrix( vec_Uij, m );

  mpfi_init( C );
  mpfi_init( Z );
}

inline void
clear
(
 vector<mpfi_ptr> &vec_Ti,
 vector<mpfi_ptr> &vec_Ui,
 vector<vector<mpfi_ptr>> &vec_Uij,
 mpfi_ptr &C,
 mpfi_ptr &Z,
 vector<vector<mpfi_ptr>> &rmatrix,
 vector<mpfi_ptr> &rdiag_sqrt,
 mpfr_ptr &mpfr_tmp,
 mpfi_ptr &mpfi_tmp,
 mpfi_ptr &mpfi_tmp2
 )
{
  mpfr_clear( mpfr_tmp );
  mpfi_clear( mpfi_tmp );
  mpfi_clear( mpfi_tmp2 );

  mpfi_clear_matrix( rmatrix );
  mpfi_clear_vector( rdiag_sqrt );

  mpfi_clear_vector( vec_Ti );
  mpfi_clear_vector( vec_Ui );
  mpfi_clear_matrix( vec_Uij );

  mpfi_clear( C );
  mpfi_clear( Z );
}

inline void
recompute
(
 size_t i_current,
 size_t m,
 unsigned int upper_bound,
 vector<int> &vec_x,
 int &LB,
 vector<int> &vec_UB,
 vector<mpfi_ptr> &vec_Ti,
 vector<mpfi_ptr> &vec_Ui,
 vector<vector<mpfi_ptr>> &vec_Uij,
 mpfi_ptr &C,
 mpfi_ptr &Z,
 const vector<vector<int>> &qfmatrix,
 vector<vector<mpfi_ptr>> &rmatrix,
 vector<mpfi_ptr> &rdiag_sqrt,
 mpfi_ptr &mpfi_tmp,
 mpfr_ptr &mpfr_tmp,
 mp_prec_t precision
 )
{
  while ( true )
    {
      precision += precision_increment;

      // set precisions
      mpfr_set_prec( mpfr_tmp, precision );
      mpfi_set_prec( mpfi_tmp, precision );

      mpfi_set_prec_matrix( rmatrix, precision );
      mpfi_set_prec_vector( rdiag_sqrt, precision );

      mpfi_set_prec_vector( vec_Ti, precision );
      mpfi_set_prec_vector( vec_Ui, precision );
      mpfi_set_prec_matrix( vec_Uij, precision );

      mpfi_set_prec( C, precision );
      mpfi_set_prec( Z, precision );


      cholesky_decomposition( qfmatrix, rmatrix, rdiag_sqrt, mpfi_tmp );

      // step 1
      mpfi_set_ui( vec_Ti[m - 1], upper_bound );
      mpfi_set_ui( vec_Ui[m - 1], 0 );
      for ( size_t j = 0; j < m; ++j )
	mpfi_set_ui( vec_Uij[j][m - 1], 0 );


      // step 2
      if( !step_2( m - 1, vec_x, LB, vec_UB, Z, vec_Ti, vec_Ui, vec_Uij, rmatrix, rdiag_sqrt, mpfi_tmp, mpfr_tmp, false ) )
	continue;

      // we make use of overflow in the for loop
      size_t i;
      for ( i = m - 2; i >= i_current && i < m - 1; --i )
	{
	  // step 5
      
	  // U_i = \sum_{j = i + 1}^m q_ij x_j
	  mpfi_set_si( vec_Ui[i], 0 );
	  for ( size_t j = i + 1; j < m; ++j )
	    mpfi_add( vec_Ui[i], vec_Ui[i], vec_Uij[i][j] );

	  // T_i = T_{i + 1} - q_{i + 1, i + 1} * (x_{i + 1} + U_{i + 1})^2
	  mpfi_add_si( mpfi_tmp, vec_Ui[i + 1], vec_x[i + 1] );
	  mpfi_sqr( mpfi_tmp, mpfi_tmp );
	  mpfi_mul( mpfi_tmp, rmatrix[i + 1][i + 1], mpfi_tmp );
	  mpfi_sub( vec_Ti[i], vec_Ti[i + 1], mpfi_tmp );

	  // step 2
	  if( !step_2( i, vec_x, LB, vec_UB, Z, vec_Ti, vec_Ui, vec_Uij, rmatrix, rdiag_sqrt, mpfi_tmp, mpfr_tmp, false ) )
	    {
	      i = m - 1;
	      break;
	    }
	}
      if ( i == m - 1 )
	continue;

      break;
    }
}

inline
bool
step_2
(
 size_t i,
 vector<int> &vec_x,
 int &LB,
 vector<int> &vec_UB,
 mpfi_ptr &Z,
 vector<mpfi_ptr> &vec_Ti,
 vector<mpfi_ptr> &vec_Ui,
 vector<vector<mpfi_ptr>> &vec_Uij,
 vector<vector<mpfi_ptr>> &rmatrix,
 vector<mpfi_ptr> &rdiag_sqrt,
 mpfi_ptr &mpfi_tmp,
 mpfr_ptr &mpfr_tmp,
 bool set_xi
 )
{
  // Z = (T_i / q_ii)^(1/2)
  mpfi_sqrt( mpfi_tmp, vec_Ti[i] );
  mpfi_div( Z, mpfi_tmp, rdiag_sqrt[i] );

  // UB_i = floor( Z - U_i )
  mpfi_sub( mpfi_tmp, Z, vec_Ui[i] );
  if ( !mpfi_get_unique_floor_si( vec_UB[i], mpfi_tmp, mpfr_tmp ) )
    return false;

  // set_xi is only false if constants are recomuted, which happens rarely
  // this justifies to compute LB in all cases so initialization
  // can later access LB

  // x_i = ceil( - Z - U_i ) - 1
  mpfi_add( mpfi_tmp, Z, vec_Ui[i] );
  mpfi_neg( mpfi_tmp, mpfi_tmp );
  mpfi_sub_si( mpfi_tmp, mpfi_tmp, 1 );
  if ( !mpfi_get_unique_ceil_si( LB, mpfi_tmp, mpfr_tmp ) )
    return false;
  if ( set_xi )
    vec_x[i] = LB;

  // U_ij = q_ij x_j
  for ( size_t j = 0; j < i; ++j )
    mpfi_mul_si( vec_Uij[j][i], rmatrix[j][i], vec_x[i] );

  return true;
}

inline
bool
mpfi_get_unique_si
(
 int &si,
 mpfi_ptr srcptr,
 mpfr_ptr mpfr_tmp 
 )
{
  mpfi_get_left( mpfr_tmp, srcptr );
  mpfr_ceil( mpfr_tmp, mpfr_tmp );
  si = mpfr_get_si( mpfr_tmp, MPFR_RNDD );

  mpfi_get_right( mpfr_tmp, srcptr );
  mpfr_floor( mpfr_tmp, mpfr_tmp );
  if ( si != mpfr_get_si( mpfr_tmp, MPFR_RNDU ) )
    return false;

  return true;
}

inline
bool
mpfi_get_unique_floor_si
(
 int &si,
 mpfi_ptr srcptr,
 mpfr_ptr mpfr_tmp
 )
{
  mpfi_get_left( mpfr_tmp, srcptr );
  mpfr_floor( mpfr_tmp, mpfr_tmp );
  si = mpfr_get_si( mpfr_tmp, MPFR_RNDD );

  mpfi_get_right( mpfr_tmp, srcptr );
  mpfr_floor( mpfr_tmp, mpfr_tmp );
  if ( si != mpfr_get_si( mpfr_tmp, MPFR_RNDU ) )
    return false;

  return true;
}

inline
bool
mpfi_get_unique_ceil_si
(
 int &si,
 mpfi_ptr srcptr,
 mpfr_ptr mpfr_tmp
 )
{
  mpfi_get_left( mpfr_tmp, srcptr );
  mpfr_ceil( mpfr_tmp, mpfr_tmp );
  si = mpfr_get_si( mpfr_tmp, MPFR_RNDD );

  mpfi_get_right( mpfr_tmp, srcptr );
  mpfr_ceil( mpfr_tmp, mpfr_tmp );
  if ( si != mpfr_get_si( mpfr_tmp, MPFR_RNDU ) )
    return false;

  return true;
}

inline
void
mpfi_init_vector
(
 vector<mpfi_ptr> &vec,
 size_t length )
{
  for ( size_t i = 0; i < length; ++i )
    {
      mpfi_ptr tmp = new __mpfi_struct;
      mpfi_init( tmp );
      vec.push_back( tmp );
    }
}

inline
void
mpfi_init_matrix
(
 vector<vector<mpfi_ptr>> &mat,
 size_t size )
{
  for ( size_t i = 0; i < size; ++i )
    {
      auto row = vector<mpfi_ptr>();
      mpfi_init_vector( row, size );
      mat.push_back( row );
    }
}

inline
void
mpfi_clear_vector
(
 vector<mpfi_ptr>& vec
)
{
  for ( auto &ptr : vec )
    {
      mpfi_clear( ptr );
      ptr = nullptr;
    }
}

inline
void
mpfi_clear_matrix
(
 vector<vector<mpfi_ptr>>& mat
)
{
  for ( auto &row : mat )
    mpfi_clear_vector( row );
}

inline
void
mpfi_set_prec_vector
(
 vector<mpfi_ptr> &vec,
 mp_prec_t prec )
{
  for ( auto &ptr : vec )
    mpfi_set_prec( ptr, prec );
}

inline
void
mpfi_set_prec_matrix
(
 vector<vector<mpfi_ptr>>& mat,
 mp_prec_t prec
)
{
  for ( auto &row : mat )
    mpfi_set_prec_vector( row, prec );
}
