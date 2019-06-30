package com.wolfram.links.rlink.dataTypes.inTypes;

/**
 * 
 *  RLinkJ source code (c) 2011-2012, Wolfram Research, Inc. 
 *  
 *  
 *  
 *   This file is part of RLinkJ interface to JRI Java library.
 *
 *   RLinkJ is free software: you can redistribute it and/or modify
 *   it under the terms of the GNU Lesser General Public License as 
 *   published by the Free Software Foundation, either version 2 of 
 *   the License, or (at your option) any later version.
 *
 *   RLinkJ is distributed in the hope that it will be useful,
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *   GNU General Public License for more details.
 *
 *   You should have received a copy of the GNU Lesser General Public 
 *   License along with RLinkJ. If not, see <http://www.gnu.org/licenses/>.
 *  
 * 
 * 
 *
 * @author Leonid Shifrin
 *
 */

import com.wolfram.links.rlink.RExecutor;

public class RVectorInType extends RInTypeImpl {

	private final int[] missingElementPositions;

	protected RVectorInType(int[] missingElementPositions,
			IRInAttributes attributes) {
		super(attributes);
		this.missingElementPositions = missingElementPositions;
	}

	@Override
	public boolean rPut(String rVar, RExecutor exec) {
		for (int pos : this.missingElementPositions) {
			if (!exec
					.eval(rVar + "[" + new Integer(pos).toString() + "] <- NA")) {
				return false;
			}
		}
		return super.rPut(rVar, exec);
	}
}
