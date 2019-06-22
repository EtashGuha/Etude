// See http://quirkygba.blogspot.com/2009/11/setting-environment-variables-in-java.html

/**
 * 
 *  RLinkJ source code (c) 2011-2012, Wolfram Research, Inc. 
 *  
 *  
 *  
 *   This file is part of RLinkJ interface to JRI Java library. It 
 *   has been derived from code published here:
 *   
 *    http://quirkygba.blogspot.com/2009/11/setting-environment-variables-in-java.html
 *    
 *   It has been trivially modified to incorporate the case for MacOSX.
 *   
 *   The license below applies unless the original code license conflicts
 *   with it, in which case this code goes under the original license. 
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

package com.wolfram.links.rlink;

import com.sun.jna.Library;
import com.sun.jna.Native;

public class Environment {
    
    public interface WinLibC extends Library {
        public int _putenv(String name);
    }
    public interface LinuxLibC extends Library {
        public int setenv(String name, String value, int overwrite);
        public int unsetenv(String name);
    }
    
    static public class POSIX {
        
        static Object libc;
        static {
          if (System.getProperty("os.name").equals("Linux")||System.getProperty("os.name").equals("Mac OS X")) {
            libc = Native.loadLibrary("c", LinuxLibC.class);
          } else {
            libc = Native.loadLibrary("msvcrt", WinLibC.class);
          }
        }

        public int setenv(String name, String value, int overwrite) {
          if (libc instanceof LinuxLibC) {
            return ((LinuxLibC)libc).setenv(name, value, overwrite);
          }
          else {
            return ((WinLibC)libc)._putenv(name + "=" + value);
          }
        }

        public int unsetenv(String name) {
          if (libc instanceof LinuxLibC) {
            return ((LinuxLibC)libc).unsetenv(name);
          }
          else {
            return ((WinLibC)libc)._putenv(name + "=");
          }
        }
    }

    public static POSIX libc = new POSIX();
}