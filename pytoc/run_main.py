import set_options
import declarations
from BuildFile import Forwards_Method

opts = set_options.options()
arch = declarations.Declare_Architecture(opts)
file = Forwards_Method.Euler_Compiler(arch[0],arch[1],arch[2],opts)


#print(arch)

