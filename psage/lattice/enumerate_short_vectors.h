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

#ifndef __ENUMERATE_SHORT_VECTORS_H
#define __ENUMERATE_SHORT_VECTORS_H

#include <vector>
#include <utility>
#include <tuple>

#include "mpfr.h"

std::tuple< std::vector<std::vector<mpfr_ptr>>, std::vector<mpfr_ptr> >
cholesky_decomposition( const std::vector<std::vector<int>>&, mpfr_prec_t );

void
enumerate_short_vectors( const std::vector<std::vector<int>>&, unsigned int, unsigned int, std::vector<std::pair<std::vector<int>, unsigned int>>& );

inline void
init_Z_UB_x( size_t, mpfr_ptr, mpfr_ptr, std::vector<mpfr_ptr>&, std::vector<mpfr_ptr>&, std::vector<mpfr_ptr>&, std::vector<mpfr_ptr>&, std::vector<mpfr_ptr>&, mpfr_ptr );

#endif
