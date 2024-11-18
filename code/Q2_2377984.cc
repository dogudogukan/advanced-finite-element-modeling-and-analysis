////////////////////////////////////////////////////////////////////////////////////////////
//                                                                                        //
//                     ME 413 INTRODUCTION TO FINITE ELEMENT ANALYSIS                     //
//                                      PROJECT - Q2                                      //
//                                                                                        //
//      Written By  : Dogukan Dogu - 2377984 - dogukan.dogu@metu.edu.tr                   //
//      Last Update : 02.07.2023                                                          //
//      Version     : 21                                                                  //
//                                                                                        //
////////////////////////////////////////////////////////////////////////////////////////////
//                                                                                        //
//                  -- OVERVIEW of the KEY FEATURES and LIMITATIONS --                    //
//                                                                                        //
// - This code solves the potential flow equation:                                        //
//                                                                                        //
//                                     del^2(psi) = 0                                     //
//                                                                                        //
//   in the 2D domain omega by using Galerkin FEM with triangular elements.               //
// - Psi denotes the stream function.                                                     //
// - The domain omega is for s2048 airfoil, and there are different options that can be   //
//   found in the program's folder. Although their names describe their general purpose,  // 
//   details of meshes can be observed using Gmsh.                                        //
// - Boundary conditions are psi = y.U on the inlet, outlet, top and the bottom of the    //
//   domain,and psi = 0 on the surface of the airfoil.                                    //
//                                                                                        //
////////////////////////////////////////////////////////////////////////////////////////////
//                                                                                        //
//                                 -- CODE USER MANUAL --                                 //
//                                                                                        //
// 1) To use the program, please create the object of "FEM_Solver_Q2" class in the        //
//    main() function with the following argument:                                        //
//    1.1) refinement_input: Refinement level for the mesh. Options: 0, 1, 2.             //
//         (enter integer)   The default value is 0.                                      //
//                                                                                        //
// 2) Call the run() command of the constituted object.                                   //
//                                                                                        //
////////////////////////////////////////////////////////////////////////////////////////////
//                                                                                        //
//                                -- s2048 MESH OPTIONS --                                //
//                                                                                        //
// All meshes below were created by using Gmsh program.                                   //
//                                                                                        //
// 1) s2048_airfoil_domain_0_refinement.msh : Mesh with 0 refinement                      //
//                                          --> 14025   elements                          //
// 2) s2048_airfoil_domain_1_refinement.msh : Mesh with 1 refinement                      //
//                                          --> 56100   elements                          //
// 3) s2048_airfoil_domain_2_refinement.msh : Mesh with 2 refinement                      //
//                                          --> 224400  elements                          //
//                                                                                        //
// The default mesh is 2048_airfoil_domain_0_refinement.msh                               //
//                                                                                        //
////////////////////////////////////////////////////////////////////////////////////////////
//                                                                                        //
//                                  -- ACKNOWLEDGMENTS --                                 //
//                                                                                        //
// I would like to express my sincere gratitude to Dr. Özgür Uğraş Baran for his guidance //
// and support, as well as for providing the opportunity to learn C++ and deal.II.        //
//                                                                                        //
////////////////////////////////////////////////////////////////////////////////////////////

//------------------------------------------------------------------------------------------
// HEADER FILES & NAMESPACES
//------------------------------------------------------------------------------------------
//--- deal.II header files
#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_in.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/manifold_lib.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_simplex_p.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/base/logstream.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/precondition.h>

//--- Other header files
#include <fstream>
#include <iostream>

//--- Namespaces
using namespace dealii;

//------------------------------------------------------------------------------------------
// CLASS - FEM_Solver_Q2
//------------------------------------------------------------------------------------------
class FEM_Solver_Q2
{
    public:
        // Constructor with default arguments
        FEM_Solver_Q2(unsigned int refinement_input = 0);

        // Make run() function public to reach and run the solver
        void run();

    private:
        // Variables
        int          gauss_quadrature_order = 2;
        unsigned int refinement;

        // Booleans
        bool is_airfoil_enabled;

        // Functions
        void import_grid();
        void print_imported_grid_info(const Triangulation<2> &triangulation);
		void setup_system();
		void assemble_system();
        void apply_boundary_conditions();
		void solve();
		void output_results() const;
        void compute_velocity();
        void compute_pressure();

        // Objects
        Triangulation<2> 	 triangulation;
        GridIn<2>            grid_in;
		FE_SimplexP<2>       fe;
		DoFHandler<2>    	 dof_handler;
		SparsityPattern      sparsity_pattern;
		SparseMatrix<double> system_LHS;
		Vector<double> 		 solution;
		Vector<double> 		 system_RHS;
        Vector<double>       velocity_x;
        Vector<double>       velocity_y;

};

//--- Constructor of the class FEM_Solver_Q2
FEM_Solver_Q2::FEM_Solver_Q2(unsigned int refinement_input)
    : refinement(refinement_input)
    , fe(1)
	, dof_handler(triangulation)
{}

//------------------------------------------------------------------------------------------
// CLASS - Right_Hand_Side --> RHS = 0
//------------------------------------------------------------------------------------------
//--- Note that this class inherits from the 'Function' class
class Right_Hand_Side : public Function<2>
{
	public:
		virtual double value(const Point<2> & p, const unsigned int component = 0) const override;
};

// Method named 'value' of the Class Right_Hand_Side
double Right_Hand_Side::value(const Point<2> &p, const unsigned int /*component*/) const
{
    // Directly return 0 as RHS is 0
    return 0;
}

//------------------------------------------------------------------------------------------
// CLASS - Inlet_Boundary_Condition --> psi = y.U where U is the freestream velocity
//------------------------------------------------------------------------------------------
//--- Note that this class inherits from the 'Function' class
class Inlet_Boundary_Condition : public Function<2>
{
public:
    virtual double value(const Point<2> &p, const unsigned int component = 0) const override;
};

// Method named 'value' of the Class Inlet_Boundary_Condition
double Inlet_Boundary_Condition::value(const Point<2> &p, const unsigned int /*component*/) const
{
    return p[1] * 1.0;
}

//------------------------------------------------------------------------------------------
// CLASS - Outlet_Boundary_Condition --> psi = y.U where U is the freestream velocity
//------------------------------------------------------------------------------------------
//--- Note that this class inherits from the 'Function' class
class Outlet_Boundary_Condition : public Function<2>
{
public:
    virtual double value(const Point<2> &p, const unsigned int component = 0) const override;
};

// Method named 'value' of the Class Outlet_Boundary_Condition
double Outlet_Boundary_Condition::value(const Point<2> &p, const unsigned int /*component*/) const
{
    return p[1] * 1.0;
}

//------------------------------------------------------------------------------------------
// CLASS - Top_Boundary_Condition --> psi = y_top.U where U is the freestream velocity
//------------------------------------------------------------------------------------------
//--- Note that this class inherits from the 'Function' class
class Top_Boundary_Condition : public Function<2>
{
public:
    virtual double value(const Point<2> &p, const unsigned int component = 0) const override;
};

// Method named 'value' of the Class Top_Boundary_Condition
double Top_Boundary_Condition::value(const Point<2> &p, const unsigned int /*component*/) const
{
    return 5 * 1.0;
}

//------------------------------------------------------------------------------------------
// CLASS - Bottom_Boundary_Condition --> psi = y_bottom.U where U is the freestream velocity
//------------------------------------------------------------------------------------------
//--- Note that this class inherits from the 'Function' class
class Bottom_Boundary_Condition : public Function<2>
{
public:
    virtual double value(const Point<2> &p, const unsigned int component = 0) const override;
};

// Method named 'value' of the Class Bottom_Boundary_Condition
double Bottom_Boundary_Condition::value(const Point<2> &p, const unsigned int /*component*/) const
{
    return (-5) * 1.0;
}

//------------------------------------------------------------------------------------------
// CLASS - Surface_Boundary_Condition --> psi = 0
//------------------------------------------------------------------------------------------
//--- Note that this class inherits from the 'Function' class
class Surface_Boundary_Condition : public Function<2>
{
public:
    virtual double value(const Point<2> &p, const unsigned int component = 0) const override;
};

// Method named 'value' of the Class Surface_Boundary_Condition
double Surface_Boundary_Condition::value(const Point<2> &p, const unsigned int /*component*/) const
{
    return 0.0;
}

//------------------------------------------------------------------------------------------
// CLASS - Calculate_Velocity
//------------------------------------------------------------------------------------------
class Calculate_Velocity : public DataPostprocessorVector<2>
{
    public:
        Calculate_Velocity() : DataPostprocessorVector<2> ("velocity_field", update_gradients)
        {}
    
        virtual void evaluate_scalar_field (const DataPostprocessorInputs::Scalar<2> &input_data,
                                            std::vector<Vector<double> > &computed_quantities) const override
        {
            AssertDimension (input_data.solution_gradients.size(), computed_quantities.size());
        
            for (unsigned int p=0; p<input_data.solution_gradients.size(); ++p)
            {
                
                AssertDimension (computed_quantities[p].size(), 2);
                 
                const double del_psi_dx = input_data.solution_gradients[p][0];
                const double del_psi_dy = input_data.solution_gradients[p][1];
            
                computed_quantities[p][0] =  del_psi_dy;  
                computed_quantities[p][1] = -del_psi_dx;   
            }
        }
};

//------------------------------------------------------------------------------------------
// CLASS - Calculate_Pressure
//------------------------------------------------------------------------------------------
class Calculate_Pressure : public DataPostprocessorVector<2>
{
    public:
        Calculate_Pressure() : DataPostprocessorVector<2> ("pressure_field", update_gradients)
        {}
    
        virtual void evaluate_scalar_field (const DataPostprocessorInputs::Scalar<2> &input_data,
                                            std::vector<Vector<double> > &computed_quantities) const override
        {
            AssertDimension (input_data.solution_gradients.size(), computed_quantities.size());
        
            for (unsigned int p=0; p<input_data.solution_gradients.size(); ++p)
            {
                
                AssertDimension(computed_quantities[p].size(), 2);

                const double del_psi_dx = input_data.solution_gradients[p][0];
                const double del_psi_dy = input_data.solution_gradients[p][1];

                const double velocity_x =  del_psi_dy;  
                const double velocity_y = -del_psi_dx;  

                const double velocity_squared = velocity_x * velocity_x + velocity_y * velocity_y;
                const double freestream_velocity_squared = 1 * 1;

                const double rho = 1.225;

                const double pressure = 0.5 * rho * freestream_velocity_squared - 0.5 * rho * velocity_squared;

                computed_quantities[p][0] = pressure; 
            }
        }
};

//------------------------------------------------------------------------------------------
// SUBFUNCTIONS
//------------------------------------------------------------------------------------------
//--- FUNCTION import_grid()
void FEM_Solver_Q2::import_grid()
{
    // Start the timer for measuring the importing time
    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

    // Attach the grid to the triangulation object
    grid_in.attach_triangulation(triangulation);

    // Open the mesh file
    std::ifstream mesh_file;
    if (refinement == 1)
    {
        mesh_file.open("s2048_airfoil_domain_1_refinement.msh");
    }
    else if (refinement == 2)
    {
        mesh_file.open("s2048_airfoil_domain_2_refinement.msh");
    }
    else
    {
        mesh_file.open("s2048_airfoil_domain_0_refinement.msh");
    }

    // Read the mesh from "s2048_airfoil_domain.msh" file which was created in Gmsh
    grid_in.read_msh(mesh_file);

    // Create the output file of the imported mesh for visual purposes
    std::string mesh_type = ("airfoil");
    std::ofstream out("mesh_for_" + mesh_type + "_" + std::to_string(triangulation.n_active_cells()) + "_active_cells.svg");
    
    // Writ the imported grid to the SVG file
    GridOut grid_out;
    grid_out.write_svg(triangulation, out);

    // Close the output file
    out.close();

    // Stop the timer
    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();

    // Calculate the duration in milliseconds
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count();

    // Print the duration
    std::cout << "Grid importing time: " << duration << " milliseconds" << std::endl;
}

//--- FUNCTION print_imported_grid_info()
void FEM_Solver_Q2::print_imported_grid_info(const Triangulation<2> &triangulation)
{
    // Print the mesh data
    std::cout << "\nIMPORTED MESH INFORMATION:"                                 << std::endl
              << "Total number of cells: "  << triangulation.n_cells()        << std::endl
			  << "Number of active cells: " << triangulation.n_active_cells() << std::endl
              << "Mesh Type: Airfoil"                                         << std::endl;
    
    {
        std::map<types::boundary_id, unsigned int> boundary_count;
        for (const auto &face : triangulation.active_face_iterators())
            if (face->at_boundary())
            boundary_count[face->boundary_id()]++;
        
        std::cout << "Boundary indicators: ";
        for (const std::pair<const types::boundary_id, unsigned int> &pair : boundary_count)
        {
            std::cout << "\n" << pair.first << '(' << pair.second << " times)";
        }
        std::cout << std::endl;
    }
}


//--- FUNCTION setup_system()
void FEM_Solver_Q2::setup_system()
{
	dof_handler.distribute_dofs(fe);
	
	std::cout << "Number of degrees of freedom: " << dof_handler.n_dofs()
			  << std::endl;
	
	DynamicSparsityPattern dynamic_sparsity_pattern(dof_handler.n_dofs());
	DoFTools::make_sparsity_pattern(dof_handler, dynamic_sparsity_pattern);
	sparsity_pattern.copy_from(dynamic_sparsity_pattern);
	
	system_LHS.reinit(sparsity_pattern);
	
	solution.reinit(dof_handler.n_dofs());
	system_RHS.reinit(dof_handler.n_dofs());
}


//--- FUNCTION assemble_system()
void FEM_Solver_Q2::assemble_system()
{
    QGauss<2> quadrature_formula(gauss_quadrature_order);

	Right_Hand_Side right_hand_side;

	FEValues<2> fe_values(fe, quadrature_formula, update_values            | update_gradients |
								                  update_quadrature_points | update_JxW_values);

	const unsigned int dofs_per_cell = fe.n_dofs_per_cell();

	FullMatrix<double> cell_LHS(dofs_per_cell, dofs_per_cell);    // Define LHS matrix for a cell
	Vector<double> cell_RHS(dofs_per_cell);                       // Define RHS vector for a cell

	std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

	for (const auto &cell : dof_handler.active_cell_iterators())
	{
		fe_values.reinit(cell);
		cell_LHS = 0;
		cell_RHS = 0;

		for (const unsigned int q_index : fe_values.quadrature_point_indices())
		{
			for (const unsigned int i : fe_values.dof_indices())
			{
				
                // Calculate LHS matrix for a cell
                for (const unsigned int j : fe_values.dof_indices())
                {
                    cell_LHS(i, j) += (fe_values.shape_grad(i, q_index) *    // grad phi_i(x_q)
					fe_values.shape_grad(j, q_index) *                       // grad phi_j(x_q)
					fe_values.JxW(q_index));			                     // dx
                }

                // Calculate RHS vector for a cell
				const auto &x_q = fe_values.quadrature_point(q_index);
				cell_RHS(i) += (fe_values.shape_value(i, q_index) *    // phi_i(x_q)
								right_hand_side.value(x_q) *		   // f(x_q)
								fe_values.JxW(q_index));			   // dx
			}
		}

		cell->get_dof_indices(local_dof_indices);
		for (const unsigned int i : fe_values.dof_indices())
		{
			// Constitute LHS matrix for the system
            for (const unsigned int j : fe_values.dof_indices())
            {
                system_LHS.add(local_dof_indices[i], local_dof_indices[j], cell_LHS(i, j));
            }

            // Constitute RHS vector for the system
			system_RHS(local_dof_indices[i]) += cell_RHS(i);
		}
	}

    // Treatment of boundary conditions
	std::map<types::global_dof_index, double> boundary_values;

	VectorTools::interpolate_boundary_values(dof_handler, 10, Inlet_Boundary_Condition()  , boundary_values);
	VectorTools::interpolate_boundary_values(dof_handler, 20, Outlet_Boundary_Condition() , boundary_values);
    VectorTools::interpolate_boundary_values(dof_handler, 30, Top_Boundary_Condition()    , boundary_values);
	VectorTools::interpolate_boundary_values(dof_handler, 40, Bottom_Boundary_Condition() , boundary_values);
    VectorTools::interpolate_boundary_values(dof_handler, 50, Surface_Boundary_Condition(), boundary_values);

	MatrixTools::apply_boundary_values(boundary_values, system_LHS, solution, system_RHS);
}


//--- FUNCTION solve()
void FEM_Solver_Q2::solve()
{
    // Start the timer for measuring the matrix solution time
    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

	SolverControl            solver_control(1000, 1e-12);
	SolverCG<Vector<double>> solver(solver_control);
	solver.solve(system_LHS, solution, system_RHS, PreconditionIdentity());
	
	std::cout << solver_control.last_step() << " CG iterations needed to obtain convergence" 
              << std::endl;
    std::cout << "\n";

    // Stop the timer
    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();

    // Calculate the duration in milliseconds
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count();

    // Print the duration
    std::cout << "Matrix solution time: " << duration << " milliseconds" << std::endl;
}

//--- FUNCTION output_results()
void FEM_Solver_Q2::output_results() const
{

    Calculate_Velocity velocity_field;
    Calculate_Pressure pressure_field;
 
    DataOut<2> data_out;
    
    data_out.attach_dof_handler (dof_handler);

    data_out.add_data_vector (solution, "stream_functions");
    data_out.add_data_vector (solution, velocity_field);
    data_out.add_data_vector (solution, pressure_field);
    data_out.build_patches ();
    
    std::string filename = "solution_for_" + std::to_string(triangulation.n_active_cells()) + "_active_cells.vtk";
    std::ofstream output(filename);
    data_out.write_vtk(output);

    // Print the filename to the terminal
    std::cout << "Solution complete!\nYou can find the corresponding solution in the\n" << filename << std::endl;
}

//--- FUNCTION run()
void FEM_Solver_Q2::run()
{
    // Start the timer for measuring the run time
    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

    // Print the welcome lines
	std::cout << "\n"
              << "-----------------------------------------------------------------\n"
              << "----------------- ME 431 Finite Element Method ------------------\n"
              << "------------------------- Term Project --------------------------\n"
              << "-------------------------- Question 2 ---------------------------\n"
              << "-----------------------------------------------------------------\n"
              << "--------------------------- Welcome! ----------------------------\n"
              << "------ This program was prepared by Dogukan Dogu, 2377984 -------\n"
              << "-----------------------------------------------------------------\n"
              << "\n"
              << "The solution of the problem given in the second question started."
			  << std::endl;

    // Apply the Finite Element Method
    import_grid();
    print_imported_grid_info(triangulation);
    setup_system();
    assemble_system();
	solve();
	output_results();

    // Stop the timer
    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();

    // Calculate the duration in milliseconds
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count();

    // Print the duration
    std::cout << "Total run time: " << duration << " milliseconds" << std::endl;

    // Print the ending lines
    std::cout << "\n"
              << "-----------------------------------------------------------------\n"
              << "--------------------- End of the Solution! ----------------------\n"
              << "-----------------------------------------------------------------\n"
              << std::endl;
}

//------------------------------------------------------------------------------------------
// MAIN FUNCTION - main()
//------------------------------------------------------------------------------------------
int main()
{
    // Default (mesh with no refinement)
    {
        FEM_Solver_Q2 Q_2_potential_flow_problem_s2048_airfoil;
        Q_2_potential_flow_problem_s2048_airfoil.run();
    }

    // Meshes with refinements
    // {
    //     FEM_Solver_Q2 Q_2_potential_flow_problem_s2048_airfoil(1);
    //     Q_2_potential_flow_problem_s2048_airfoil.run();
    // }
    // {
    //     FEM_Solver_Q2 Q_2_potential_flow_problem_s2048_airfoil(2);
    //     Q_2_potential_flow_problem_s2048_airfoil.run();
    // }

    return 0;
}