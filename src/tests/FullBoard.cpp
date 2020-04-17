/*
    This file is part of Leela Zero.
    Copyright (C) 2017-2019 Gian-Carlo Pascutto and contributors

    Leela Zero is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    Leela Zero is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with Leela Zero.  If not, see <http://www.gnu.org/licenses/>.

    Additional permission under GNU GPL version 3 section 7

    If you modify this Program, or any covered work, by linking or
    combining it with NVIDIA Corporation's libraries from the
    NVIDIA CUDA Toolkit and/or the NVIDIA CUDA Deep Neural
    Network library and/or the NVIDIA TensorRT inference library
    (or a modified version of those libraries), containing parts covered
    by the terms of the respective license agreement, the licensors of
    this Program grant you additional permission to convey the resulting
    work.
*/

#include "config.h"

#include <array>
#include <cassert>

#include "FullBoard.h"
#include "Network.h"
#include "Utils.h"
#include "Zobrist.h"

using namespace Utils;


template<class Function>
std::uint64_t FullBoard::calc_hash(int komove, Function transform) const {
    auto res = Zobrist::zobrist_empty;

    for (auto i = 0; i < m_numvertices; i++) {
        if (m_state[i] != INVAL) {
            res ^= Zobrist::zobrist[m_state[i]][transform(i)];
        }
    }

    if (m_tomove == BLACK) {
        res ^= Zobrist::zobrist_blacktomove;
    }

    res ^= Zobrist::zobrist_ko[transform(komove)];

    return res;
}

std::uint64_t FullBoard::calc_hash(int komove) const {
    return calc_hash(komove, [](const auto vertex) { return vertex; });
}

std::uint64_t FullBoard::calc_symmetry_hash(int komove, int symmetry) const {
    return calc_hash(komove, [this, symmetry](const auto vertex) {
        if (vertex == NO_VERTEX) {
            return NO_VERTEX;
        } else {
            const auto newvtx = Network::get_symmetry(get_xy(vertex), symmetry, m_boardsize);
            return get_vertex(newvtx.first, newvtx.second);
        }
    });
}

std::uint64_t FullBoard::get_hash() const {
    return m_hash;
}

void FullBoard::set_to_move(int tomove) {
    if (m_tomove != tomove) {
        m_hash ^= Zobrist::zobrist_blacktomove;
    }
    FastBoard::set_to_move(tomove);
}

/// 落子并更新hash表
int FullBoard::update_board(const int color, const int i) {
    assert(i != FastBoard::PASS);
    assert(m_state[i] == EMPTY);

    m_hash ^= Zobrist::zobrist[m_state[i]][i];
    set_state(i, vertex_t(color));
    update_continue_info(i, vertex_t(color));
    m_hash ^= Zobrist::zobrist[m_state[i]][i];

    // No ko
    return NO_VERTEX;
}

void FullBoard::display_board(int lastmove) {
    FastBoard::display_board(lastmove);

    myprintf("Hash: %llX\n\n", get_hash());
}

void FullBoard::reset_board(int size) {
    FastBoard::reset_board(size);

    m_hash = calc_hash();
}
