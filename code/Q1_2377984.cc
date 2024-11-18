////////////////////////////////////////////////////////////////////////////////////////////
//                                                                                        //
//                     ME 413 INTRODUCTION TO FINITE ELEMENT ANALYSIS                     //
//                                      PROJECT - Q1                                      //
//                                                                                        //
//      Written By  : Dogukan Dogu - 2377984 - dogukan.dogu@metu.edu.tr                   //
//      Last Update : 02.07.2023                                                          //
//      Version     : 13                                                                  //
//                                                                                        //
////////////////////////////////////////////////////////////////////////////////////////////
//                                                                                        //
//                  -- OVERVIEW of the KEY FEATURES and LIMITATIONS --                    //
//                                                                                        //
// - This code solves the heat transfer equation:                                         //
//                                                                                        //
//                                 - del^2(T) = r^2 . theta                               //
//                                                                                        //
//   in the provided 2D domain omgea by using Galerkin FEM with quadrilateral elements.   //
// - The domain omgea consists of the area between two nested half circles with radii     //
//   of 0.25 and 2.00, respectively.                                                      //
// - Boundary conditions are T=0 on the outer curved surface, T=50 on the inner curved    //
//   surface, and delT/deln=0 on the sides (no flux on sides).                            //
//                                                                                        //
////////////////////////////////////////////////////////////////////////////////////////////
//                                                                                        //
//                                 -- CODE USER MANUAL --                                 //
//                                                                                        //
// 1) To use the program, please create the object of "FEM_Solver_Q1" class with          //
//    arguments in given order in the main() function:                                    //
//    1.1) basic_number_of_cells_input: Basic number of cells for the mesh generation.    //
//         (enter integer)              In other words, number of active cells without    //
//                                      any refinement. The default value is 2.           //
//    1.2) refinement_level_input: Level of mesh refinement. It uses refine_global()      //
//         (enter integer)         function of the deal.II library. Basically, it         //
//                                 increases the number of active cells to:               //       
//                                         "basic_number_of_cells_input x                 //
//                                          2 ^ (dim x refinement_level_input)"           //
//                                 Note that, in this program, dim = 2 as it is for 2D.   //
//                                 The default value is 0.                                //
//    1.3) gauss_quadrature_order_input:  The order of the Gauss quadrature integration   //
//         (enter integer)                formula used for numerical integration. Note    //
//                                        that NGP point integration can evaluate:        //
//                                                         "2 x NGP − 1"                  //
//                                        order polynomial functions exactly.             //
//                                        The default value is 2.                         //
//    1.4) is_renumbering_dofs_enabled_input: Boolean flag to specify whether the degrees //
//         (enter true/false)                 of freedom (DOFs) are renumbered using the  //
//                                            Cuthill-McKee algorithm. The default value  //
//                                            is false.                                   //
//    1.5) is_gnu_plot_enabled_input: Boolean flag that specifies whether GNU Plot is     //
//         (enter true/false)         enabled for visualizing the mesh and nodes. It may  //
//                                    not so useful for high number of cells since node   // 
//                                    numbers might overlap. The default value is false.  //
//                                                                                        //
// 2) Call the run() command of the constituted object.                                   //
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
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
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
#include <cmath>

//--- Namespaces
using namespace dealii;

//------------------------------------------------------------------------------------------
// CLASS - FEM_Solver_Q1
//------------------------------------------------------------------------------------------
class FEM_Solver_Q1
{
	public:
        // Constructor with default arguments
		FEM_Solver_Q1(unsigned int basic_number_of_cells_input       = 2, 
                      unsigned int refinement_level_input            = 0, 
                      unsigned int gauss_quadrature_order_input      = 2,
                      bool         is_renumbering_dofs_enabled_input = false,
                      bool         is_gnu_plot_enabled_input         = false);
        
        // Make run() function public to reach and run the solver
		void run();
	
	private:
        // Variables
        unsigned int basic_number_of_cells;
        unsigned int refinement_level;
        unsigned int gauss_quadrature_order;

        // Booleans
        bool         is_renumbering_dofs_enabled;
        bool         is_gnu_plot_enabled;

        // Functions
		void generate_mesh();
		void setup_system();
        void setup_system_renumbering_dofs();
        void gnu_plot();
		void assemble_system();
        void apply_boundary_conditions();
		void solve();
		void output_results() const;
		
        // Objects
		Triangulation<2> 	 triangulation;
		FE_Q<2>          	 fe;
		DoFHandler<2>    	 dof_handler;
		SparsityPattern      sparsity_pattern;
		SparseMatrix<double> system_LHS;
		Vector<double> 		 solution;
		Vector<double> 		 system_RHS;
};

//--- Constructor of the class FEM_Solver_Q1
FEM_Solver_Q1::FEM_Solver_Q1(unsigned int basic_number_of_cells_input, 
                             unsigned int refinement_level_input, 
                             unsigned int gauss_quadrature_order_input,
                             bool         is_renumbering_dofs_enabled_input,
                             bool         is_gnu_plot_enabled_input)
	: basic_number_of_cells(basic_number_of_cells_input)
    , refinement_level(refinement_level_input)
    , gauss_quadrature_order(gauss_quadrature_order_input)
    , is_renumbering_dofs_enabled(is_renumbering_dofs_enabled_input)
    , is_gnu_plot_enabled(is_gnu_plot_enabled_input)
    , fe(1)
	, dof_handler(triangulation)
{}

//------------------------------------------------------------------------------------------
// CLASS - Right_Hand_Side --> RHS = r^2 * theta
//------------------------------------------------------------------------------------------
//--- Note that this class inherits from the 'Function' class
class Right_Hand_Side : public Function<2>
{
	public:
		virtual double value(const Point<2> &p, const unsigned int component = 0) const override;
};

//--- Method named 'value' of the class Right_Hand_Side
double Right_Hand_Side::value(const Point<2> &p, const unsigned int /*component*/) const
{
    // Calculate the radius
    double r = std::sqrt(p[0] * p[0] + p[1] * p[1]);

    // To deal with the origin point
    if (r == 0)
    {
        return 0;
    }

    // Calculate the angle
    double theta = std::atan2(p[1], p[0]);

    return r * r * theta;
}

//------------------------------------------------------------------------------------------
// CLASS - Inner_Boundary_Condition --> Gamma_1 ---> T = 50
//------------------------------------------------------------------------------------------
//--- Note that this class inherits from the 'Function' class
class Inner_Boundary_Condition : public Function<2>
{
public:
    virtual double value(const Point<2> &p, const unsigned int component = 0) const override;
};

//--- Method named 'value' of the Class Inner_Boundary_Condition
double Inner_Boundary_Condition::value(const Point<2> &p, const unsigned int /*component*/) const
{
    return 50.0;
}

//------------------------------------------------------------------------------------------
// CLASS - Outer_Boundary_Condition --> Gamma_2 ---> T = 0
//------------------------------------------------------------------------------------------
//--- Note that this class inherits from the 'Function' class
class Outer_Boundary_Condition : public Function<2>
{
public:
    virtual double value(const Point<2> &p, const unsigned int component = 0) const override;
};

//--- Method named 'value' of the Class Outer_Boundary_Condition
double Outer_Boundary_Condition::value(const Point<2> &p, const unsigned int /*component*/) const
{
    return 0.0;
}

//------------------------------------------------------------------------------------------
// SUBFUNCTIONS
//------------------------------------------------------------------------------------------
//--- FUNCTION generate_mesh()
void FEM_Solver_Q1::generate_mesh()
{
	// Point object and other variables
	Point<2> center(0, 0);
	double   inner_radius = 0.25;
	double   outer_radius = 2;
    bool     colorize     = true;

	// GridGenerator
	GridGenerator::half_hyper_shell(triangulation, 
                                    center, 
                                    inner_radius, 
                                    outer_radius, 
                                    basic_number_of_cells, 
                                    colorize);

    // Rotate the domain to obtain desired result
    GridTools::rotate(dealii::numbers::PI/2.0, triangulation);

	// Refinement
    if (refinement_level >= 1)
    {
        triangulation.refine_global(refinement_level);
    }

	// Print the mesh data
	std::cout << "\nINPUTS:\n"
              << "Basic number of cells: "  << basic_number_of_cells       << std::endl
              << "Refinement level: "       << refinement_level            << std::endl
              << "Gauss quadrature order: " << gauss_quadrature_order      << std::endl
              << "Renumbering dofs: "       << std::boolalpha 
                                            << is_renumbering_dofs_enabled << std::endl
              << "GNU Plot: "               << std::boolalpha 
                                            << is_gnu_plot_enabled         << std::endl                                            
              << "\n----------------------------------------------------------------\n" << std::endl;
	std::cout << "Meshing was completed!\n"
              << "The generated mesh can be found in mesh_with_" 
               + std::to_string(triangulation.n_active_cells()) + "_cells.svg file.\n"  << std::endl;
    std::cout << "OUTPUTS:\n"
              << "Total number of cells: "  << triangulation.n_cells()        << std::endl
			  << "Number of active cells: " << triangulation.n_active_cells() << std::endl;

	// Assign boundary IDs in accordance with Gamma_1, Gamma_2, Gamma_3
    const int inner_boundary_ID = 1;
    const int outer_boundary_ID = 2;
    const int side_boundary_ID  = 3;

	for (const auto &cell : triangulation.active_cell_iterators())
	{
		for (const auto &face : cell->face_iterators())
		{
			if (face->at_boundary())
			{
				if (face->center().distance(center) < inner_radius + 1e-6)
                {
                    // Inner boundary
					face->set_boundary_id(inner_boundary_ID);
                }
                else if (std::abs(face->center()[1]) < 1e-6)
                {
                    // Side boundary
					face->set_boundary_id(side_boundary_ID);
                }
				else
                {
                    // Outer boundary
					face->set_boundary_id(outer_boundary_ID);
                }
			}
		}
	}

	// Output the mesh file as ---> mesh_for_(triangulation.n_active_cells())_cells.svg
	std::ofstream out_1("mesh_for_" + std::to_string(triangulation.n_active_cells()) + "_active_cells.svg");
    GridOut grid_out_1;
    grid_out_1.write_svg(triangulation, out_1);
}

//--- FUNCTION gnu_plot()
void FEM_Solver_Q1::gnu_plot()
{
    // File mesh_with_nodes_for_(triangulation.n_cells())_cells.gpl
    std::ofstream out_3("mesh_with_nodes_for_" + std::to_string(triangulation.n_active_cells()) + "_active_cells.gpl");

    out_3 << "set xlabel 'X'" << std::endl;  // Set X-axis label
    out_3 << "set ylabel 'Y'" << std::endl;  // Set Y-axis label

    // Set plot title
    out_3 << "set title 'Mesh with Nodes for Number of Active Cells = " << triangulation.n_active_cells()
          << std::endl;

    out_3 << "set style line 1 lt 1 lc rgb 'green' lw 2" << std::endl;          // Define line style
    out_3 << "set style line 2 lt 7 lc rgb 'red' lw 1 pt 7 ps 1" << std::endl;  // Define point style

    out_3 << "plot '-' using 1:2 with lines ls 1 notitle, "
          << "'-' with labels font ',10' textcolor rgb 'black' offset 0.5,0.5 point ls 2 notitle" << std::endl;

    GridOut().write_gnuplot(triangulation, out_3);
    out_3 << "e" << std::endl;

    std::map<types::global_dof_index, Point<2>> support_points;
    DoFTools::map_dofs_to_support_points(MappingQ1<2>(), dof_handler, support_points);

    // Modify the write_gnuplot_dof_support_point_info function
    out_3 << "# Support Point Info" << std::endl;
    out_3 << "# X Y NodeNumber" << std::endl;

    for (const auto& point : support_points) 
    {
        const Point<2>& p = point.second;
        const types::global_dof_index& node_number = point.first;
        out_3 << p[0] << " " << p[1] << " " << node_number << std::endl;
    }

    out_3 << "e" << std::endl;

    out_3.close();

    // Open the generated gnuplot file in gnuplot directly
    system(("gnuplot -p mesh_with_nodes_for_" + std::to_string(triangulation.n_active_cells()) + "_active_cells.gpl").c_str());

}

//--- FUNCTION setup_system()
void FEM_Solver_Q1::setup_system()
{
	dof_handler.distribute_dofs(fe);
	
	std::cout << "Number of degrees of freedom: " << dof_handler.n_dofs()
			  << std::endl;
	
	DynamicSparsityPattern dynamic_sparsity_pattern(dof_handler.n_dofs());
	DoFTools::make_sparsity_pattern(dof_handler, dynamic_sparsity_pattern);
	sparsity_pattern.copy_from(dynamic_sparsity_pattern);
	
    std::ofstream out("sparsity_pattern_for_" + std::to_string(triangulation.n_active_cells()) + "_active_cells.svg");
    sparsity_pattern.print_svg(out);

	system_LHS.reinit(sparsity_pattern);
	
	solution.reinit(dof_handler.n_dofs());
	system_RHS.reinit(dof_handler.n_dofs());
}

//--- FUNCTION setup_system_renumbering_dofs()
void FEM_Solver_Q1::setup_system_renumbering_dofs()
{
    dof_handler.distribute_dofs(fe);
	
	std::cout << "Number of degrees of freedom: " << dof_handler.n_dofs()
			  << std::endl;

    DoFRenumbering::Cuthill_McKee(dof_handler);
    
    DynamicSparsityPattern dynamic_sparsity_pattern(dof_handler.n_dofs());
    DoFTools::make_sparsity_pattern(dof_handler, dynamic_sparsity_pattern);
    sparsity_pattern.copy_from(dynamic_sparsity_pattern);
    
    std::ofstream out("renumbered_sparsity_pattern_for_" + std::to_string(triangulation.n_active_cells()) + "_active_cells.svg");
    sparsity_pattern.print_svg(out);

    system_LHS.reinit(sparsity_pattern);
        
    solution.reinit(dof_handler.n_dofs());
    system_RHS.reinit(dof_handler.n_dofs());
}

//--- FUNCTION assemble_system()
void FEM_Solver_Q1::assemble_system()
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

	std::map<types::global_dof_index, double> boundary_values;

	VectorTools::interpolate_boundary_values(dof_handler, 1, Inner_Boundary_Condition(), boundary_values);
	VectorTools::interpolate_boundary_values(dof_handler, 2, Outer_Boundary_Condition(), boundary_values);
	
    MatrixTools::apply_boundary_values(boundary_values, system_LHS, solution, system_RHS);
}

//--- FUNCTION solve()
void FEM_Solver_Q1::solve()
{
    // Start the timer for measuring the matrix solution time
    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

	SolverControl            solver_control(1000, 1e-12);
	SolverCG<Vector<double>> solver(solver_control);
	solver.solve(system_LHS, solution, system_RHS, PreconditionIdentity());
    
	
	std::cout << solver_control.last_step() << " CG iterations needed to obtain convergence" 
              << std::endl;

    // Stop the timer
    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();

    // Calculate the duration in milliseconds
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count();

    // Print the duration
    std::cout << "Matrix solution time: " << duration << " milliseconds" << std::endl;
}
 
//--- FUNCTION output_results()
void FEM_Solver_Q1::output_results() const
{
	DataOut<2> data_out;
	
	data_out.attach_dof_handler(dof_handler);
	data_out.add_data_vector(solution, "solution");
	
	data_out.build_patches();
	
	std::string filename = "solution_for_" + std::to_string(triangulation.n_active_cells()) + "_active_cells_and_" +
                           std::to_string(gauss_quadrature_order) + "_gauss_quadrature_order.vtk";
    std::ofstream output(filename);
    data_out.write_vtk(output);

    // Print the solutions of node values - for part 3
    const unsigned int N = solution.size();    // Number of rows
    std::cout << "Solutions of Node Values:" << std::endl;
    for (unsigned int i = 0; i < N; ++i)
    {
        const unsigned int index = i;
        std::cout << "T_" << i << " = " << solution[index] << "\t"
                  << std::endl;
    }

    std::cout << "\n";

       // Print the filename to the terminal
    std::cout << "Solution complete!\nYou can find the corresponding solution in the\n" << filename << std::endl;
}

//--- FUNCTION run()
void FEM_Solver_Q1::run()
{
    // Start the timer for measuring the run time
    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

    // Print the welcome lines
	std::cout << "\n"
              << "----------------------------------------------------------------\n"
              << "----------------- ME 431 Finite Element Method -----------------\n"
              << "------------------------- Term Project -------------------------\n"
              << "-------------------------- Question 1 --------------------------\n"
              << "----------------------------------------------------------------\n"
              << "--------------------------- Welcome! ---------------------------\n"
              << "------ This program was prepared by Dogukan Dogu, 2377984 ------\n"
              << "----------------------------------------------------------------\n"
              << "\n"
              << "The solution of the problem given in the first question started."
			  << std::endl;
	
    // Apply the Finite Element Method
	generate_mesh();

    if (is_renumbering_dofs_enabled == true)
    {
        setup_system_renumbering_dofs();
    }
    else
    {
        setup_system();
    }

    if (is_gnu_plot_enabled == true)
    {
        gnu_plot();
    }

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
              << "----------------------------------------------------------------\n"
              << "--------------------- End of the Solution! ---------------------\n"
              << "----------------------------------------------------------------\n"
              << std::endl;
}

//------------------------------------------------------------------------------------------
// MAIN FUNCTION - main()
//------------------------------------------------------------------------------------------
int main()
{
    // A reasonable solution is set by default
    // {
    //     FEM_Solver_Q1 Q_1_heat_transfer_problem(2,2,3,false,true); 
    //     Q_1_heat_transfer_problem.run();
    // }
    
    // Part 2
    // {
    //     FEM_Solver_Q1 Q_1_heat_transfer_problem(5,0,3,false,true); 
    //     Q_1_heat_transfer_problem.run();
    // }
    // {
    //     FEM_Solver_Q1 Q_1_heat_transfer_problem(5,1,3,false,true); 
    //     Q_1_heat_transfer_problem.run();
    // }
    // {
    //     FEM_Solver_Q1 Q_1_heat_transfer_problem(2,2,3,false,true); 
    //     Q_1_heat_transfer_problem.run();
    // }
    // {
    //     FEM_Solver_Q1 Q_1_heat_transfer_problem(5,3,3,false,false); 
    //     Q_1_heat_transfer_problem.run();
    // }
    // {
    //     FEM_Solver_Q1 Q_1_heat_transfer_problem(5,4,3,false,false); 
    //     Q_1_heat_transfer_problem.run();
    // }
    // {
    //    FEM_Solver_Q1 Q_1_heat_transfer_problem(5,5,3,true,false); 
    //    Q_1_heat_transfer_problem.run();
    // }

    // Part 3
    // {
    //    FEM_Solver_Q1 Q_1_heat_transfer_problem(2,2,1,false,false); 
    //    Q_1_heat_transfer_problem.run();
    // }
    // {
    //    FEM_Solver_Q1 Q_1_heat_transfer_problem(2,2,1,false,false); 
    //    Q_1_heat_transfer_problem.run();
    // }
    // {
    //    FEM_Solver_Q1 Q_1_heat_transfer_problem(2,2,1,false,false); 
    //    Q_1_heat_transfer_problem.run();
    // }

    // Part 4
    // {
    //    FEM_Solver_Q1 Q_1_heat_transfer_problem(5,5,3,false,false); 
    //    Q_1_heat_transfer_problem.run();
    // }
    // {
    //    FEM_Solver_Q1 Q_1_heat_transfer_problem(5,5,3,true,false); 
    //    Q_1_heat_transfer_problem.run();
    // }
    // {
    //    FEM_Solver_Q1 Q_1_heat_transfer_problem(2,2,3,false,false); 
    //    Q_1_heat_transfer_problem.run();
    // }
    // {
    //    FEM_Solver_Q1 Q_1_heat_transfer_problem(2,2,3,true,false); 
    //    Q_1_heat_transfer_problem.run();
    // }

    return 0;
}